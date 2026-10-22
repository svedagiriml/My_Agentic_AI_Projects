"""MCP toolbox used by every subagent.

* Resolves each MCP server either in-process (tests/local) or over Streamable HTTP
  (docker-compose / Cloud Run) with service-to-service auth.
* Opens one session per server per agent run (reused for all ReAct steps).
* Every call is wrapped with: allow-list check -> timeout -> retry -> circuit breaker
  -> metrics/tracing -> output guardrail (tool output is untrusted data).
"""
from __future__ import annotations

import importlib
import json
import time
from contextlib import AsyncExitStack
from typing import Any

from loanflow.common.config import get_settings
from loanflow.common.exceptions import ForbiddenError, ToolError, UpstreamUnavailable
from loanflow.common.logging import get_logger
from loanflow.common.schemas import ToolCallRecord
from loanflow.mcp_layer.compat import connect, parse_tool_result, tool_schema
from loanflow.observability import tracing
from loanflow.observability.metrics import GUARDRAIL_EVENTS, TOOL_LATENCY
from loanflow.resilience import get_breaker, retry_async, with_timeout
from loanflow.security.guardrails import detect_injection, redact_pii

log = get_logger(__name__)

SERVER_MODULES = {
    "document": "loanflow.mcp_servers.document_server",
    "credit_bureau": "loanflow.mcp_servers.credit_bureau_server",
    "risk_engine": "loanflow.mcp_servers.risk_engine_server",
    "compliance": "loanflow.mcp_servers.compliance_server",
    "reporting": "loanflow.mcp_servers.reporting_server",
}


def _auth_headers(url: str) -> dict[str, str]:
    s = get_settings()
    if s.use_gcp_id_tokens:
        from google.auth.transport.requests import Request
        from google.oauth2 import id_token

        audience = url.split("/mcp")[0]
        return {"Authorization": f"Bearer {id_token.fetch_id_token(Request(), audience)}"}
    return {"Authorization": f"Bearer {s.internal_service_token}"}


def _target(server: str) -> tuple[Any, dict[str, str] | None]:
    s = get_settings()
    if s.transport_mode == "http":
        base = s.mcp_urls.get(server)
        if not base:
            raise UpstreamUnavailable(f"MCP_URLS has no entry for '{server}'")
        url = base.rstrip("/") + "/mcp"
        return url, _auth_headers(url)
    return importlib.import_module(SERVER_MODULES[server]).mcp, None


class MCPToolbox:
    def __init__(self, allowed: dict[str, list[str]]):
        """allowed = {server_name: [tool_name, ...]} — the agent's tool allow-list."""
        self.allowed = allowed
        self._stack = AsyncExitStack()
        self._sessions: dict[str, Any] = {}
        self._catalog: list[dict[str, Any]] = []
        self._tool_server: dict[str, str] = {}
        self.records: list[ToolCallRecord] = []

    async def __aenter__(self) -> MCPToolbox:
        s = get_settings()
        for server, tools in self.allowed.items():
            target, headers = _target(server)
            breaker = get_breaker(f"mcp:{server}", s.cb_failure_threshold, s.cb_reset_timeout_s)

            async def open_session(target=target, headers=headers):
                return await self._stack.enter_async_context(connect(target, headers, s.tool_timeout_s))

            session = await breaker.call(
                lambda server=server, open_session=open_session: with_timeout(open_session(), s.tool_timeout_s, f"connect {server}"))
            self._sessions[server] = session
            listed = await with_timeout(session.list_tools(), s.tool_timeout_s, f"list_tools {server}")
            for t in listed.tools:
                if t.name in tools:
                    schema = tool_schema(t)
                    schema["server"] = server
                    self._catalog.append(schema)
                    self._tool_server[t.name] = server
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    @property
    def catalog(self) -> list[dict[str, Any]]:
        return self._catalog

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        server = self._tool_server.get(tool)
        if server is None:
            raise ForbiddenError(f"tool '{tool}' is not in this agent's allow-list", details={"tool": tool})
        s = get_settings()
        session = self._sessions[server]
        breaker = get_breaker(f"mcp:{server}", s.cb_failure_threshold, s.cb_reset_timeout_s)
        t0 = time.perf_counter()
        ok, err, payload = False, None, None
        try:
            with tracing.span(f"tool:{tool}", input=arguments, metadata={"server": server}) as sp:
                result = await breaker.call(
                    lambda: retry_async(
                        lambda: with_timeout(session.call_tool(tool, arguments), s.tool_timeout_s, f"{server}.{tool}"),
                        attempts=s.max_retries,
                        name=f"{server}.{tool}",
                    )
                )
                is_error, payload = parse_tool_result(result)
                if is_error:
                    raise ToolError(f"{tool} protocol error: {str(payload)[:300]}")
                if not isinstance(payload, dict):
                    payload = {"ok": True, "value": payload}
                self._guard_output(tool, payload)
                ok = bool(payload.get("ok", True))
                if not ok:
                    err = json.dumps(payload.get("error"))
                sp.update(output=payload)
            return payload
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency = (time.perf_counter() - t0) * 1000
            TOOL_LATENCY.labels(server, tool, str(ok)).observe(latency / 1000)
            preview = redact_pii(json.dumps(payload, default=str)).text[:300] if payload is not None else None
            self.records.append(ToolCallRecord(tool=tool, server=server, arguments=arguments, ok=ok,
                                               latency_ms=round(latency, 2), error=err, output_preview=preview))

    @staticmethod
    def _guard_output(tool: str, payload: dict[str, Any]) -> None:
        """Tool output is untrusted: flag injected directives in any string field (doc text is handled upstream)."""
        for key, val in payload.items():
            if isinstance(val, str) and key != "content" and detect_injection(val).is_injection:
                GUARDRAIL_EVENTS.labels("tool_output_injection").inc()
                payload[key] = "[REMOVED: suspected injected instruction]"
