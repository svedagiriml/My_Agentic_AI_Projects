"""Supervisor API (public entry point, behind Cloud Load Balancer + Cloud Armor WAF).

Customer endpoints:   POST /v1/chat, GET /v1/chat/{thread_id}, DELETE /v1/me/memory
Staff (HITL) console: GET /v1/hitl/pending, POST /v1/hitl/{thread_id}/decision
Ops:                  GET /healthz, /readyz, /metrics, POST /v1/auth/dev-token (non-prod only)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel, Field

from bankcore.config import Settings, get_settings
from bankcore.errors import AuthorizationError, DependencyTimeout, NotFound, ValidationFailed
from bankcore.guardrails.pii import redact_text
from bankcore.http import harden
from bankcore.logging import configure_logging, get_logger, trace_id_var
from bankcore.observability import tracing
from bankcore.observability.cost import UsageCallback
from bankcore.schemas.common import AgentName, RiskLevel, Role
from bankcore.schemas.messages import (
    ApprovalDecision, ChatRequest, ChatResponse, PendingApprovalView, SubAgentResponse, TaskSummary,
)
from bankcore.security.auth import Principal, issue_token, require_customer, require_staff

from supervisor.app.container import Container, build_container

log = get_logger(__name__)


def _thread_cfg(thread_id: str, callbacks: list | None = None) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60, "callbacks": callbacks or []}


async def _build_response(container: Container, thread_id: str, values: dict[str, Any], interrupted: bool,
                          cached: bool) -> ChatResponse:
    results = {k: SubAgentResponse.model_validate(v) for k, v in (values.get("results") or {}).items()}
    tasks = [TaskSummary(task_id=k, agent=r.agent, status=r.status, tools_used=[c.tool for c in r.tool_calls])
             for k, r in results.items()]
    if interrupted and values.get("pending"):
        action = values["pending"]["action"]
        done = [r.answer for k, r in results.items() if r.status == "completed" and r.answer]
        reply = "\n\n".join(done + ["For your security, this request needs approval from a bank officer before it "
                                    "can be completed. We'll update you as soon as it's reviewed."])
        reply = redact_text(reply).text
        return ChatResponse(status="pending_approval", reply=reply, session_id=values["session_id"],
                            thread_id=thread_id, trace_id=values["trace_id"], tasks=tasks, cached_route=cached,
                            pending_approval=PendingApprovalView(action_id=action["action_id"],
                                                                 agent=AgentName(action["agent"]), tool=action["tool"],
                                                                 summary=action["summary_for_reviewer"],
                                                                 risk_level=RiskLevel(action["risk_level"])))
    status = values.get("status", "completed")
    if status == "running":
        status = "completed"
    return ChatResponse(status=status, reply=values.get("final") or "", session_id=values["session_id"],
                        thread_id=thread_id, trace_id=values["trace_id"], tasks=tasks, cached_route=cached)


def create_app(container: Container | None = None, settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    holder: dict[str, Container | None] = {"c": container}
    thread_owner: dict[str, str] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        configure_logging("supervisor", s.log_level)
        if holder["c"] is None:
            holder["c"] = await build_container(s)
        yield
        tracing.flush()
        if container is None and holder["c"] is not None:
            await holder["c"].aclose()

    app = harden(FastAPI(title="Horizon Trust Bank — Agentic Customer Support", version="1.0.0", lifespan=lifespan),
                 "supervisor")
    app.add_middleware(CORSMiddleware, allow_origins=["https://app.horizontrust.example"], allow_methods=["GET", "POST",
                       "DELETE"], allow_headers=["authorization", "content-type", "x-request-id"])

    def c() -> Container:
        assert holder["c"] is not None
        return holder["c"]

    async def _remember_owner(thread_id: str, user_id: str) -> None:
        thread_owner[thread_id] = user_id
        r = c().redis
        if r is not None:
            try:
                await r.set(f"thread:{thread_id}:owner", user_id, ex=7 * 86400)
            except Exception:  # noqa: BLE001
                pass

    async def _owner(thread_id: str) -> str | None:
        r = c().redis
        if r is not None:
            try:
                v = await r.get(f"thread:{thread_id}:owner")
                if v:
                    return v.decode() if isinstance(v, bytes) else v
            except Exception:  # noqa: BLE001
                pass
        return thread_owner.get(thread_id)

    # ---------------------------------------------------------------- ops --
    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        checks = {"redis": False, "long_term_memory": c().long_term is not None, "llm": c().deps.llm is not None}
        if c().redis is not None:
            try:
                checks["redis"] = bool(await c().redis.ping())
            except Exception:  # noqa: BLE001
                pass
        return {"status": "ready", "checks": checks, "degraded": not all(checks.values())}

    class DevTokenRequest(BaseModel):
        subject: str = Field(pattern=r"^(USR-\d{4,8}|STF-\d{3,6})$")
        role: Role = Role.CUSTOMER

    @app.post("/v1/auth/dev-token", include_in_schema=False)
    async def dev_token(body: DevTokenRequest):
        if s.is_prod:
            raise NotFound("not available")
        scopes = ["chat"] if body.role == Role.CUSTOMER else ["hitl:approve", "hitl:read"]
        return {"access_token": issue_token(body.subject, body.role, scopes, settings=s), "token_type": "bearer"}

    # ------------------------------------------------------------- chat ----
    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest, request: Request, p: Principal = Depends(require_customer)) -> ChatResponse:
        ct = c()
        ip = request.client.host if request.client else "unknown"
        await ct.rate_limiter.hit(f"user:{p.subject}:m", limit=s.rate_limit_per_minute, window_s=60)
        await ct.rate_limiter.hit(f"user:{p.subject}:d", limit=s.rate_limit_per_day, window_s=86400)
        await ct.rate_limiter.hit(f"ip:{ip}", limit=s.rate_limit_per_minute * 3, window_s=60)

        session_id = body.session_id or uuid.uuid4().hex[:24]
        await ct.deps.memory.st.bind_owner(session_id, p.subject)
        thread_id = f"{session_id}-{uuid.uuid4().hex[:8]}"
        trace_id = tracing.new_trace_id()
        trace_id_var.set(trace_id)
        await _remember_owner(thread_id, p.subject)

        # PII is redacted at ingress: raw identifiers never reach LLMs, caches, logs or checkpoints.
        clean = redact_text(body.message).text
        usage = UsageCallback("supervisor", "turn")
        cfg = _thread_cfg(thread_id, [usage, *tracing.langfuse_callbacks(trace_id)])
        init = {"user_id": p.subject, "session_id": session_id, "thread_id": thread_id, "trace_id": trace_id,
                "channel": body.channel, "message": clean, "deadline_ts": time.time() + s.request_deadline_s,
                "results": {}, "trajectory": [], "hitl_decisions": [], "pending": None, "status": "running"}
        try:
            await asyncio.wait_for(ct.graph.ainvoke(init, config=cfg), timeout=s.request_deadline_s)
        except asyncio.TimeoutError as e:
            raise DependencyTimeout("supervisor deadline exceeded") from e
        snap = await ct.graph.aget_state(_thread_cfg(thread_id))
        resp = await _build_response(ct, thread_id, snap.values, bool(snap.next), snap.values.get("plan_source") ==
                                     "cache")
        log.info("chat turn", status=resp.status, tasks=[t.agent.value for t in resp.tasks],
                 tokens_in=usage.usage.input_tokens, tokens_out=usage.usage.output_tokens,
                 cost_usd=round(usage.cost_usd, 6), plan_source=snap.values.get("plan_source"))
        return resp

    @app.get("/v1/chat/{thread_id}", response_model=ChatResponse)
    async def chat_status(thread_id: str, p: Principal = Depends(require_customer)) -> ChatResponse:
        if await _owner(thread_id) != p.subject:
            raise NotFound("thread not found")
        snap = await c().graph.aget_state(_thread_cfg(thread_id))
        if not snap.values:
            raise NotFound("thread not found")
        return await _build_response(c(), thread_id, snap.values, bool(snap.next), False)

    @app.delete("/v1/me/memory")
    async def forget_me(p: Principal = Depends(require_customer)):
        """Right to erasure (GDPR Art.17 / CCPA / India DPDP Act s.12)."""
        deleted = await c().long_term.forget_user(p.subject) if c().long_term else 0
        return {"deleted_long_term_records": deleted}

    # -------------------------------------------------------------- HITL ---
    @app.get("/v1/hitl/pending")
    async def hitl_pending(_staff: Principal = Depends(require_staff)):
        return {"items": await c().deps.hitl.list()}

    @app.post("/v1/hitl/{thread_id}/decision", response_model=ChatResponse)
    async def hitl_decide(thread_id: str, decision: ApprovalDecision,
                          staff: Principal = Depends(require_staff)) -> ChatResponse:
        ct = c()
        item = await ct.deps.hitl.get(thread_id)
        if item is None:
            raise NotFound("no pending approval for this thread")
        if item["user_id"] == staff.subject:
            raise AuthorizationError("four-eyes principle: you cannot approve your own request")
        if time.time() > item.get("expires_at", float("inf")) and decision.approved:
            raise ValidationFailed("approval window expired", user_message="This approval request has expired.")
        snap = await ct.graph.aget_state(_thread_cfg(thread_id))
        if not snap.next:
            raise NotFound("workflow is not waiting for approval")
        trace_id_var.set(snap.values.get("trace_id"))
        cfg = _thread_cfg(thread_id, tracing.langfuse_callbacks(snap.values.get("trace_id", "")))
        await ct.graph.ainvoke(Command(resume={"approved": decision.approved, "reviewer_id": staff.subject,
                                               "note": redact_text(decision.note).text}), config=cfg)
        snap = await ct.graph.aget_state(_thread_cfg(thread_id))
        log.info("hitl decision applied", thread_id=thread_id, approved=decision.approved, reviewer=staff.subject)
        return await _build_response(ct, thread_id, snap.values, bool(snap.next), False)

    return app


app = create_app()
