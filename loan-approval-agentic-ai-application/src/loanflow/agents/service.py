"""Subagent microservice (one Cloud Run service per agent).

  AGENT_NAME=check_credit uvicorn loanflow.agents.service:app --port 8080

Endpoints
  POST /v1/tasks   signed AgentTask JSON  ->  signed AgentResult JSON
  GET  /v1/card    agent card (name, description, tools, output schema) - discovery
  GET  /healthz /readyz /metrics

Every inbound task is verified (schema + checksum + HMAC) and every outbound
result is signed: message verification at both sides of the handoff.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from loanflow.agents.specs import AGENTS
from loanflow.common.config import get_settings
from loanflow.common.exceptions import AgentExecutionError, LoanFlowError, ValidationFailure
from loanflow.common.integrity import parse_and_verify, sign_message
from loanflow.common.logging import configure_logging, get_logger, run_id_var
from loanflow.common.schemas import AgentResult, AgentTask, ErrorInfo
from loanflow.observability import tracing
from loanflow.observability.metrics import NODE_LATENCY
from loanflow.security.auth import Principal, require_service
from loanflow.security.rate_limit import check_rate_limit

log = get_logger(__name__)


async def execute_task(task: AgentTask) -> AgentResult:
    """Run one subagent task. Never raises: failures become a signed `failed` AgentResult."""
    agent = AGENTS.get(task.recipient)
    run_id_var.set(task.run_id)
    if agent is None:
        err = ErrorInfo(code="UNKNOWN_AGENT", message=task.recipient)
        return sign_message(AgentResult(run_id=task.run_id, in_reply_to=task.message_id, sender=task.recipient,
                                        status="failed", error=err))
    payload: dict[str, Any] = task.model_dump(mode="json")
    payload["notes"] = (task.context or {}).get("_notes", [])
    try:
        with NODE_LATENCY.labels(agent.name).time():
            out = await asyncio.wait_for(agent.run(payload), timeout=task.deadline_s)
        result = AgentResult(
            run_id=task.run_id, in_reply_to=task.message_id, sender=agent.name,
            status="needs_review" if out.needs_review else "success",
            output=out.output, reasoning_trace=out.trace, tool_calls=out.tool_calls, metrics=out.metrics,
        )
    except TimeoutError:
        result = AgentResult(run_id=task.run_id, in_reply_to=task.message_id, sender=agent.name, status="failed",
                             error=ErrorInfo(code="AGENT_TIMEOUT", message=f"exceeded {task.deadline_s}s", retryable=True))
    except LoanFlowError as exc:
        log.warning("agent failed", extra={"extra_fields": {"agent": agent.name, **exc.to_dict()}})
        result = AgentResult(run_id=task.run_id, in_reply_to=task.message_id, sender=agent.name, status="failed",
                             error=ErrorInfo(code=exc.code, message=exc.message[:500], retryable=exc.retryable),
                             reasoning_trace=getattr(exc, "trace", []), tool_calls=getattr(exc, "tool_records", []))
    except Exception as exc:  # noqa: BLE001 - last line of defence
        log.exception("agent crashed")
        err = AgentExecutionError(f"{type(exc).__name__}: {exc}")
        result = AgentResult(run_id=task.run_id, in_reply_to=task.message_id, sender=agent.name, status="failed",
                             error=ErrorInfo(code=err.code, message=err.message[:500], retryable=True))
    if result.status == "needs_review" and result.output is None:  # pragma: no cover - guarded by schema
        raise ValidationFailure("needs_review without output")
    return sign_message(result)


def create_agent_app(agent_name: str) -> FastAPI:
    from loanflow.common.http import install_common

    if agent_name not in AGENTS:
        raise RuntimeError(f"AGENT_NAME must be one of {sorted(AGENTS)}")
    configure_logging(get_settings().log_level)
    agent = AGENTS[agent_name]
    app = FastAPI(title=f"loanflow-agent-{agent_name}", version="1.0.0", docs_url=None, redoc_url=None)
    install_common(app, f"agent-{agent_name}")

    @app.get("/v1/card")
    async def card(_: Principal = Depends(require_service)):
        return {"name": agent.name, "description": agent.description, "pattern": "ReAct",
                "tools": agent.tools, "required_tools": agent.required_tools,
                "output_schema": agent.output_model.model_json_schema()}

    @app.post("/v1/tasks")
    async def run_task(request: Request, caller: Principal = Depends(require_service)):
        await check_rate_limit(caller.sub, limit_per_minute=600, scope=f"agent:{agent_name}")
        task = parse_and_verify(AgentTask, await request.body())
        if task.recipient != agent_name:
            raise ValidationFailure(f"task addressed to {task.recipient}, this is {agent_name}")
        with tracing.span(f"service:{agent_name}", input={"run_id": task.run_id}):
            result = await execute_task(task)
        tracing.flush()
        return JSONResponse(content=result.model_dump(mode="json"))

    return app


app = create_agent_app(os.environ["AGENT_NAME"]) if os.environ.get("AGENT_NAME") else None
