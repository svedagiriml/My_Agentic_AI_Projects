"""Generic ReAct sub-agent microservice.

The same image is deployed eight times as separate Cloud Run services; ``AGENT_NAME`` picks
the AgentSpec (instructions + MCP tool allow-list) and ``MCP_URLS`` points it at its own MCP
server only.

    POST /v1/invoke   body: HandoffEnvelope(SubAgentRequest)  ->  HandoffEnvelope(SubAgentResponse)
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from bankcore.agents.react import ReActAgent
from bankcore.agents.specs import get_spec
from bankcore.config import Settings, get_settings
from bankcore.errors import BankAgentError, DependencyTimeout
from bankcore.http import harden
from bankcore.llm.factory import ResilientLLM
from bankcore.logging import configure_logging, get_logger, trace_id_var
from bankcore.mcp.client import MCPToolClient
from bankcore.observability import tracing
from bankcore.resilience.bulkhead import bulkhead, remaining
from bankcore.schemas.common import ErrorInfo
from bankcore.schemas.messages import HandoffEnvelope, SubAgentRequest, SubAgentResponse
from bankcore.security.service_auth import verify_service_request
from bankcore.security.signing import open_envelope, seal_envelope

log = get_logger(__name__)


def build_agent(settings: Settings) -> ReActAgent:
    spec = get_spec(settings.agent_name)
    mcp = MCPToolClient(agent=spec.name.value, domain=spec.mcp_domain, target=settings.mcp_urls[spec.name.value],
                        allowed_tools=set(spec.tools), settings=settings)
    return ReActAgent(spec, ResilientLLM("subagent", settings), mcp, settings=settings)


def create_app(agent: ReActAgent | None = None, settings: Settings | None = None,
               verify_caller: bool = True) -> FastAPI:
    s = settings or get_settings()
    state: dict = {"agent": agent}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        configure_logging(f"agent-{s.agent_name}", s.log_level)
        if state["agent"] is None:
            state["agent"] = build_agent(s)
        yield
        tracing.flush()

    app = harden(FastAPI(title=f"agent-{s.agent_name}", lifespan=lifespan, docs_url=None, redoc_url=None),
                 f"agent-{s.agent_name}")

    if verify_caller:
        @app.middleware("http")
        async def _service_auth(request: Request, call_next):
            # authenticate BEFORE body parsing so unauthenticated callers learn nothing about the schema
            if request.url.path.startswith("/v1/"):
                try:
                    verify_service_request(request, audience=f"agent-{s.agent_name}", allowed_callers={"supervisor"},
                                           settings=s)
                except BankAgentError as e:
                    return JSONResponse({"error": e.to_dict()}, status_code=e.http_status)
            return await call_next(request)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "agent": s.agent_name}

    @app.post("/v1/invoke", response_model=HandoffEnvelope)
    async def invoke(env: HandoffEnvelope, request: Request) -> HandoffEnvelope:
        me = f"agent-{s.agent_name}"
        req = open_envelope(env, expected_recipient=me, model=SubAgentRequest, allowed_senders={"supervisor"},
                            settings=s)                                  # message verification at handoff
        trace_id_var.set(env.trace_id)
        agent: ReActAgent = state["agent"]
        try:
            budget = remaining(req.deadline_ts)
            async with bulkhead(me, s.bulkhead_max_concurrency):
                resp = await asyncio.wait_for(agent.run(req, trace_id=env.trace_id), timeout=budget)
        except (asyncio.TimeoutError, DependencyTimeout):
            resp = SubAgentResponse(task_id=req.task_id, agent=agent.spec.name, status="failed",
                                    answer="This step took too long to complete.",
                                    error=ErrorInfo(code="agent_timeout", message="timeout", retryable=True))
        log.info("task finished", agent=s.agent_name, status=resp.status, tools=[t.tool for t in resp.tool_calls],
                 elapsed_s=round(time.time() - env.created_at, 2))
        return seal_envelope(resp, sender=me, recipient="supervisor", trace_id=env.trace_id,
                             thread_id=env.thread_id, settings=s)

    return app


app = create_app()
