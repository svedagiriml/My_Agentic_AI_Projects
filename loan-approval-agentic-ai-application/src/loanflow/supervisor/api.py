"""Supervisor API (public entry point, Cloud Run service `loanflow-supervisor`).

  POST /v1/loan-applications            submit {applicant_id, document_id}  -> 202 + run_id  (?wait=true for sync)
  GET  /v1/runs/{run_id}                status + final decision
  GET  /v1/runs/{run_id}/audit          full trajectory, CoT routing log, tool calls, audit trail
  GET  /v1/reviews                      pending HITL reviews            (underwriter)
  POST /v1/runs/{run_id}/review         approve / reject a referred run (underwriter)
  GET  /v1/admin/memory/quarantine      quarantined long-term memories  (admin)
  POST /internal/jobs                   Pub/Sub push / internal job intake (service auth)
  POST /v1/dev/token                    mint a JWT  (APP_ENV=local|test only)
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from loanflow.common.config import get_settings
from loanflow.common.exceptions import ForbiddenError, NotFoundError, ValidationFailure
from loanflow.common.http import install_common
from loanflow.common.logging import configure_logging, get_logger
from loanflow.common.schemas import HumanReviewDecision, LoanApplicationRequest, RunStatus, new_id
from loanflow.memory.kv import get_kv
from loanflow.memory.long_term import get_long_term_memory, seed_long_term_memory
from loanflow.memory.short_term import ShortTermMemory, remember_idempotency_key
from loanflow.observability.metrics import GUARDRAIL_EVENTS
from loanflow.security.auth import Principal, issue_token, require_roles, require_service
from loanflow.security.guardrails import check_officer_notes
from loanflow.security.rate_limit import check_rate_limit
from loanflow.supervisor.checkpoint import checkpointer_ctx
from loanflow.supervisor.graph import build_graph
from loanflow.supervisor.queue import make_queue
from loanflow.supervisor.runner import LoanOrchestrator

log = get_logger(__name__)


class AppState:
    orchestrator: LoanOrchestrator
    queue: Any


S = AppState()


async def _handle_job(job: dict[str, Any]) -> None:
    kv = get_kv()
    if job["type"] == "start":
        st = await S.orchestrator.execute(job["run_id"], job["applicant_id"], job["document_id"],
                                          job["officer"], job.get("notes"))
    elif job["type"] == "resume":
        st = await S.orchestrator.resume(job["run_id"], job["review"], job["reviewer"])
        await kv.delete(f"hitl:pending:{job['run_id']}")
    else:
        raise ValidationFailure(f"unknown job type {job['type']}")
    if st.status == "awaiting_human":
        await kv.set(f"hitl:pending:{st.run_id}", st.pending_review, ttl_s=30 * 86_400)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_logging(s.log_level)
    if not s.is_prod:
        await seed_long_term_memory()
    async with checkpointer_ctx() as cp:
        S.orchestrator = LoanOrchestrator(build_graph(checkpointer=cp))
        S.queue = make_queue()
        if s.worker_enabled:  # API-only instances set WORKER_ENABLED=false; worker service consumes the stream
            await S.queue.start(_handle_job, concurrency=s.max_concurrent_runs)
        yield
        await S.queue.stop()


app = FastAPI(title="LoanFlow Supervisor API", version="1.0.0", lifespan=lifespan)
install_common(app, "supervisor")


async def _authorize_run(run_id: str, user: Principal) -> None:
    owner = await get_kv().get(f"run_owner:{run_id}")
    if owner is None:
        raise NotFoundError(f"run {run_id} not found")
    if owner != user.sub and not {"underwriter", "admin"} & set(user.roles):
        raise ForbiddenError("not allowed to view this run")


@app.post("/v1/loan-applications", status_code=202, response_model=RunStatus)
async def submit(req: LoanApplicationRequest, wait: bool = Query(False, description="block until done/paused"),
                 user: Principal = Depends(require_roles("loan_officer", "underwriter"))):
    await check_rate_limit(user.sub)
    try:
        notes = check_officer_notes(req.notes)
    except Exception:
        GUARDRAIL_EVENTS.labels("api_notes_injection").inc()
        raise
    run_id = new_id("run")
    if req.idempotency_key:
        existing = await get_kv().get(f"idem:{user.sub}:{req.idempotency_key}")
        if existing:
            return JSONResponse(status_code=200, content=(await _status(existing)).model_dump(mode="json"))
        await remember_idempotency_key(f"{user.sub}:{req.idempotency_key}", run_id)
    await get_kv().set(f"run_owner:{run_id}", user.sub, ttl_s=90 * 86_400)
    job = {"type": "start", "run_id": run_id, "applicant_id": req.applicant_id, "document_id": req.document_id,
           "officer": user.sub, "notes": [f"officer note: {notes}"] if notes else []}
    queued = RunStatus(run_id=run_id, status="queued", applicant_id=req.applicant_id, document_id=req.document_id)
    await ShortTermMemory(run_id).save_status(queued.model_dump(mode="json"))
    if wait:
        await asyncio.wait_for(_handle_job(job), timeout=get_settings().agent_timeout_s * 10)
        return JSONResponse(status_code=200, content=(await _status(run_id)).model_dump(mode="json"))
    await S.queue.enqueue(job)
    return queued


async def _status(run_id: str) -> RunStatus:
    try:
        return await S.orchestrator.status(run_id)
    except NotFoundError:
        cached = await ShortTermMemory(run_id).status()
        if cached:
            return RunStatus.model_validate(cached)
        raise


@app.get("/v1/runs/{run_id}", response_model=RunStatus)
async def get_run(run_id: str, user: Principal = Depends(require_roles("loan_officer", "underwriter"))):
    await _authorize_run(run_id, user)
    return await _status(run_id)


@app.get("/v1/runs/{run_id}/audit")
async def get_audit(run_id: str, user: Principal = Depends(require_roles("underwriter", "loan_officer"))):
    await _authorize_run(run_id, user)
    return await S.orchestrator.audit(run_id)


@app.get("/v1/reviews")
async def pending_reviews(_: Principal = Depends(require_roles("underwriter"))):
    kv = get_kv()
    return {"pending": [await kv.get(k) for k in await kv.keys("hitl:pending:*")]}


@app.post("/v1/runs/{run_id}/review", response_model=RunStatus)
async def review(run_id: str, body: HumanReviewDecision, wait: bool = Query(True),
                 user: Principal = Depends(require_roles("underwriter"))):
    st = await _status(run_id)
    if st.status != "awaiting_human":
        raise ValidationFailure(f"run is {st.status}, not awaiting_human")
    owner = await get_kv().get(f"run_owner:{run_id}")
    if owner == user.sub:
        raise ForbiddenError("maker-checker: the submitting officer cannot review their own application")
    job = {"type": "resume", "run_id": run_id, "review": body.model_dump(), "reviewer": user.sub}
    if wait:
        await _handle_job(job)
        return await _status(run_id)
    await S.queue.enqueue(job)
    return st


@app.get("/v1/admin/memory/quarantine")
async def quarantine(_: Principal = Depends(require_roles("admin"))):
    return {"items": await get_long_term_memory().store.quarantined()}


@app.post("/internal/jobs", status_code=204)
async def internal_job(request: Request, _: Principal = Depends(require_service)):
    """Pub/Sub push endpoint (message.data = base64 JSON job)."""
    import base64
    import json

    body = await request.json()
    job = json.loads(base64.b64decode(body["message"]["data"])) if "message" in body else body
    await _handle_job(job)


class DevTokenReq(BaseModel):
    sub: str
    roles: list[str]


@app.post("/v1/dev/token")
async def dev_token(req: DevTokenReq):
    if get_settings().env not in ("local", "test"):
        raise NotFoundError("not available")
    return {"access_token": issue_token(req.sub, req.roles), "token_type": "bearer"}  # type: ignore[arg-type]
