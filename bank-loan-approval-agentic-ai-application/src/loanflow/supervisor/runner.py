"""Orchestrator: starts, resumes and inspects runs; writes outcomes to long-term memory."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from langgraph.types import Command

from loanflow.common.config import get_settings
from loanflow.common.exceptions import NotFoundError, ValidationFailure
from loanflow.common.logging import get_logger, run_id_var
from loanflow.common.schemas import ErrorInfo, FinalDecision, RunStatus
from loanflow.memory.long_term import MemoryRecord, get_long_term_memory
from loanflow.memory.short_term import ShortTermMemory
from loanflow.observability import tracing
from loanflow.observability.metrics import DECISIONS, HITL_PENDING, NODE_LATENCY, RUNS_IN_FLIGHT

log = get_logger(__name__)


class LoanOrchestrator:
    def __init__(self, graph):
        self.graph = graph
        self._sem = asyncio.Semaphore(get_settings().max_concurrent_runs)  # bulkhead per instance

    @staticmethod
    def _cfg(run_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 60}

    async def execute(self, run_id: str, applicant_id: str, document_id: str, officer: str,
                      notes: list[str] | None = None) -> RunStatus:
        initial = {"run_id": run_id, "applicant_id": applicant_id, "document_id": document_id, "officer": officer,
                   "notes": notes or [], "results": {}, "attempts": {}, "trajectory": [], "routing_log": [],
                   "audit": [], "tool_calls": [], "metrics": {}, "steps": 0, "error": None, "fatal": False,
                   "human_review": None}
        return await self._drive(run_id, initial, officer)

    async def resume(self, run_id: str, review: dict[str, Any], reviewer: str) -> RunStatus:
        st = await self.status(run_id)
        if st.status != "awaiting_human":
            raise ValidationFailure(f"run {run_id} is not awaiting human review (status={st.status})")
        return await self._drive(run_id, Command(resume={**review, "reviewer": reviewer}), reviewer)

    async def _drive(self, run_id: str, graph_input: Any, user: str) -> RunStatus:
        run_id_var.set(run_id)
        async with self._sem:
            RUNS_IN_FLIGHT.inc()
            t0 = time.perf_counter()
            try:
                with tracing.span("loan_run", input={"run_id": run_id}):
                    tracing.tag_trace(run_id=run_id, user_id=user, tags=["loan-processing"])
                    await self.graph.ainvoke(graph_input, self._cfg(run_id))
            finally:
                RUNS_IN_FLIGHT.dec()
                NODE_LATENCY.labels("run_total").observe(time.perf_counter() - t0)
                tracing.flush()
        status = await self.status(run_id)
        await ShortTermMemory(run_id).save_status(status.model_dump(mode="json"))
        if status.status == "awaiting_human":
            HITL_PENDING.inc()
        if status.decision is not None:
            if isinstance(graph_input, Command):
                HITL_PENDING.dec()
            DECISIONS.labels(status.decision.decision.value).inc()
            await self._remember(status)
        return status

    async def status(self, run_id: str) -> RunStatus:
        snap = await self.graph.aget_state(self._cfg(run_id))
        v = snap.values or {}
        if not v:
            raise NotFoundError(f"run {run_id} not found")
        results = v.get("results", {})
        final = results.get("compile_report") or results.get("compile_rejection")
        interrupts = [i.value for t in (snap.tasks or ()) for i in (getattr(t, "interrupts", ()) or ())]
        if final:
            state = "completed"
        elif interrupts:
            state = "awaiting_human"
        elif v.get("fatal"):
            state = "failed"
        else:
            state = "running" if snap.next else "failed"
        err = v.get("error")
        return RunStatus(
            run_id=run_id, status=state, applicant_id=v["applicant_id"], document_id=v["document_id"],
            current_node=(snap.next[0] if snap.next else None), trajectory=v.get("trajectory", []),
            decision=FinalDecision.model_validate(final) if final else None,
            pending_review=interrupts[0] if interrupts else None,
            error=ErrorInfo(code=err["code"], message=err.get("message", ""), retryable=bool(err.get("retryable")))
            if err and state == "failed" else None,
        )

    async def audit(self, run_id: str) -> dict[str, Any]:
        snap = await self.graph.aget_state(self._cfg(run_id))
        v = snap.values or {}
        if not v:
            raise NotFoundError(f"run {run_id} not found")
        return {k: v.get(k) for k in ("run_id", "trajectory", "routing_log", "audit", "tool_calls", "metrics",
                                      "human_review", "review_reasons")}

    async def _remember(self, status: RunStatus) -> None:
        d = status.decision
        assert d is not None
        try:
            await get_long_term_memory().remember(MemoryRecord(
                kind="episodic", namespace=f"applicant:{d.applicant_id}",
                content=f"Previous loan decision: application {d.document_id} was {d.decision.value} (risk grade {d.risk_grade}, "
                        f"score {d.credit_score}). Reasons: {'; '.join(d.reasons)[:600]}",
                source="compile_report" if d.decision.value == "APPROVED" else "compile_rejection",
                confidence=1.0, metadata={"run_id": d.run_id, "decision_id": d.decision_id}))
        except Exception as exc:  # noqa: BLE001 - memory write is best-effort
            log.warning("episodic memory write failed", extra={"extra_fields": {"error": str(exc)}})
