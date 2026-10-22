"""MCP client used by sub-agents.

STATE ISOLATION is enforced here (layer 1 of 3):
  1. allow-list   — only tools named in the agent's AgentSpec are exposed to its LLM; any
                    other tool the server advertises is dropped, and calling one raises
                    ``ToolNotPermitted`` before any network I/O.
  2. network      — each agent only knows its own MCP server URL and its service identity is
                    only accepted by that server (see BankMCPServer.build_app).
  3. token scope  — delegation tokens carry ``aud=mcp:<domain>``; other servers reject them.

Hidden parameters (delegation_token / approval_token) are stripped from the schema the LLM
sees and injected by the runtime, so the model can neither read nor forge them.
"""
from __future__ import annotations

import copy
import time
from contextlib import asynccontextmanager
from typing import Any

from pydantic import ValidationError

from bankcore.agents.policies import MUTATING_TOOLS
from bankcore.config import Settings, get_settings
from bankcore.errors import BankAgentError, DependencyError, ToolNotPermitted
from bankcore.logging import get_logger
from bankcore.mcp import compat
from bankcore.mcp.server import HIDDEN_PARAMS
from bankcore.resilience.bulkhead import with_timeout
from bankcore.resilience.circuit_breaker import get_breaker
from bankcore.resilience.retry import retry_async
from bankcore.schemas.tools import TOOL_OUTPUT_MODELS, ToolEnvelope, ToolError
from bankcore.security.service_auth import outbound_service_token

log = get_logger(__name__)


def to_llm_tool(spec: dict) -> dict:
    schema = copy.deepcopy(spec["input_schema"]) or {"type": "object", "properties": {}}
    for hidden in HIDDEN_PARAMS:
        schema.get("properties", {}).pop(hidden, None)
        if hidden in schema.get("required", []):
            schema["required"].remove(hidden)
    schema.pop("title", None)
    return {"type": "function", "function": {"name": spec["name"], "description": spec["description"],
                                             "parameters": schema}}


class MCPToolSession:
    def __init__(self, conn, agent: str, allowed: set[str], timeout_s: float, breaker_name: str) -> None:
        self._conn, self.agent, self.allowed = conn, agent, allowed
        self.timeout_s, self.breaker = timeout_s, get_breaker(breaker_name)
        self._tools: list[dict] | None = None

    async def llm_tools(self) -> list[dict]:
        if self._tools is None:
            advertised = await self.breaker.call(lambda: with_timeout(self._conn.list_tools, self.timeout_s,
                                                                      "mcp:list_tools"))
            names = {t["name"] for t in advertised}
            unexpected = names - self.allowed
            if unexpected:
                log.warning("MCP server advertised tools outside allow-list; dropping", agent=self.agent,
                            tools=sorted(unexpected))
            self._tools = [to_llm_tool(t) for t in advertised if t["name"] in self.allowed]
        return self._tools

    async def call(self, name: str, arguments: dict[str, Any], *, delegation_token: str,
                   approval_token: str | None = None) -> tuple[ToolEnvelope, float]:
        if name not in self.allowed:
            raise ToolNotPermitted(f"agent '{self.agent}' may not call tool '{name}'")
        args = {k: v for k, v in (arguments or {}).items() if k not in HIDDEN_PARAMS}
        args["delegation_token"] = delegation_token
        if approval_token:
            args["approval_token"] = approval_token
        started = time.perf_counter()

        async def _once():
            return await self.breaker.call(lambda: with_timeout(lambda: self._conn.call_tool(name, args),
                                                                self.timeout_s, f"mcp:{name}"))
        # Read-only tools are retried; mutating tools rely on server-side idempotency keys.
        is_error, structured, text = await (_once() if name in MUTATING_TOOLS else retry_async(_once, attempts=3))
        latency = (time.perf_counter() - started) * 1000
        if is_error or structured is None:
            return ToolEnvelope(ok=False, tool=name, error=ToolError(
                code="tool_transport_error", message=(text or "tool failed")[:300])), latency
        try:
            env = ToolEnvelope.model_validate(structured)
            if env.ok and env.data is not None and name in TOOL_OUTPUT_MODELS:
                TOOL_OUTPUT_MODELS[name].model_validate(env.data)      # client-side output verification
        except ValidationError as e:
            log.error("tool output failed schema verification", tool=name, errors=e.error_count())
            return ToolEnvelope(ok=False, tool=name, error=ToolError(
                code="tool_output_invalid", message="Tool returned malformed data")), latency
        return env, latency


class MCPToolClient:
    def __init__(self, *, agent: str, domain: str, target: Any, allowed_tools: set[str],
                 settings: Settings | None = None) -> None:
        self.agent, self.domain, self.target = agent, domain, target
        self.allowed = set(allowed_tools)
        self.settings = settings or get_settings()

    @asynccontextmanager
    async def session(self):
        headers = {}
        if isinstance(self.target, str):
            headers["authorization"] = "Bearer " + outbound_service_token(
                f"mcp-{self.domain}", caller=f"agent-{self.agent}", settings=self.settings, url=self.target)
        try:
            async with compat.open_client(self.target, headers=headers, timeout_s=self.settings.mcp_timeout_s) as c:
                yield MCPToolSession(c, self.agent, self.allowed, self.settings.mcp_timeout_s, f"mcp:{self.domain}")
        except BaseExceptionGroup as eg:  # anyio task groups wrap errors raised inside the session
            leaf = eg
            while isinstance(leaf, BaseExceptionGroup) and leaf.exceptions:
                leaf = leaf.exceptions[0]
            if isinstance(leaf, BankAgentError):
                raise leaf from eg
            raise DependencyError(f"MCP server {self.domain} unreachable: {leaf!r}") from eg
