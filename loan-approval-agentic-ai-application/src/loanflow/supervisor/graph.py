"""LangGraph supervisor graph.

            ┌────────────────────────── supervisor (CoT brain) ◄──────────────────────────┐
  START ──► │ decides next_node via cache -> LLM -> policy, validated by policy guard      │
            └──┬──────────┬────────────┬───────────┬────────────┬───────────┬─────────┬──┘
               ▼          ▼            ▼           ▼            ▼           ▼         ▼
        fetch_document validate_doc check_credit assess_risk check_compl. human_review compile_*
               └──────────┴────────────┴───────────┴────────────┴───────────┴─────────┘
                                       (every node returns to the supervisor)

* Every node is a subagent reached through an AgentTransport (HTTP to its own Cloud Run
  service in prod). Tasks are signed; results are verified and schema-validated.
* `human_review` uses LangGraph `interrupt()`: the run is checkpointed (Postgres) and
  the process can exit; POST /v1/runs/{id}/review resumes it later, on any instance.
* State is checkpointed after every node, so a crash mid-run (or a long DB/API wait)
  resumes from the last completed node rather than restarting.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from loanflow.common.config import get_settings
from loanflow.common.exceptions import LoanFlowError, MessageIntegrityError, ValidationFailure
from loanflow.common.integrity import sign_message, validate_node_output
from loanflow.common.logging import get_logger, run_id_var
from loanflow.common.schemas import AGENT_NODES, AgentResult, AgentTask, ErrorInfo, HumanReviewDecision, utcnow
from loanflow.memory.short_term import ShortTermMemory
from loanflow.observability import tracing
from loanflow.observability.metrics import NODE_LATENCY
from loanflow.supervisor.router import SupervisorBrain, review_triggers
from loanflow.supervisor.state import LoanState
from loanflow.supervisor.transport import AgentTransport, get_transport

log = get_logger(__name__)

# Least-privilege context: each subagent only sees the upstream outputs it needs.
NODE_INPUTS: dict[str, tuple[str, ...]] = {
    "fetch_document": (),
    "validate_document": ("fetch_document",),
    "check_credit": ("validate_document",),
    "assess_risk": ("validate_document", "check_credit"),
    "check_compliance": ("validate_document",),
    "compile_report": ("validate_document", "check_credit", "assess_risk", "check_compliance"),
    "compile_rejection": ("validate_document", "check_credit", "assess_risk", "check_compliance"),
}
FATAL_ON_ERROR = {"fetch_document", "compile_report", "compile_rejection"}


def _audit(actor: str, action: str, detail: str, **kw: Any) -> dict[str, Any]:
    return {"ts": utcnow().isoformat(), "actor": actor, "action": action, "detail": detail[:500], **kw}


def make_agent_node(name: str, transport: AgentTransport):
    async def node(state: LoanState) -> dict[str, Any]:
        s = get_settings()
        run_id_var.set(state["run_id"])
        attempt = state.get("attempts", {}).get(name, 0) + 1
        ctx: dict[str, Any] = {n: state["results"][n] for n in NODE_INPUTS[name] if n in state.get("results", {})}
        if name.startswith("compile"):
            ctx["human_review"] = state.get("human_review")
            ctx["_error"] = state.get("error")
            ctx["_audit"] = [{k: v for k, v in a.items() if k in ("ts", "actor", "action", "detail", "message_id", "checksum")}
                             for a in state.get("audit", [])][-40:]
        ctx["_notes"] = state.get("notes", [])
        task = sign_message(AgentTask(
            run_id=state["run_id"], recipient=name, applicant_id=state["applicant_id"],
            document_id=state["document_id"], instruction=state.get("instruction") or f"Perform {name}.",
            context=ctx, deadline_s=s.agent_timeout_s, attempt=attempt,
        ))
        stm = ShortTermMemory(state["run_id"])
        await stm.append_message("supervisor", {"to": name, "message_id": task.message_id, "instruction": task.instruction})

        with tracing.span(f"node:{name}", input={"attempt": attempt}), NODE_LATENCY.labels(f"graph:{name}").time():
            try:
                result = await transport.invoke(task)
                if result.in_reply_to != task.message_id:
                    raise MessageIntegrityError("reply does not correlate with task")
                validate_node_output(name, result)  # handoff verification (typed contract)
            except LoanFlowError as exc:
                result = sign_message(AgentResult(run_id=state["run_id"], in_reply_to=task.message_id, sender=name,
                                                  status="failed", error=ErrorInfo(code=exc.code, message=exc.message[:500],
                                                                                   retryable=exc.retryable)))

        await stm.append_message(name, {"status": result.status, "message_id": result.message_id,
                                        "error": result.error.model_dump() if result.error else None})
        m = dict(state.get("metrics", {}))
        for k in ("llm_calls", "prompt_tokens", "completion_tokens", "cost_usd", "tool_calls", "cache_hits"):
            m[k] = m.get(k, 0) + getattr(result.metrics, k)
        m["agent_latency_ms"] = m.get("agent_latency_ms", 0) + result.metrics.latency_ms
        update: dict[str, Any] = {
            "attempts": {name: attempt},
            "trajectory": [name],
            "tool_calls": [{"node": name, **t.model_dump(mode="json")} for t in result.tool_calls],
            "metrics": m,
            "audit": [_audit(name, f"agent_{result.status}",
                             result.error.message if result.error else f"{len(result.tool_calls)} tool calls",
                             message_id=result.message_id, checksum=result.checksum)],
        }
        if result.status in ("success", "needs_review"):
            update["results"] = {name: result.output}
            update["error"] = None
            if result.status == "needs_review":
                await stm.add_note(f"{name} flagged the application for human review")
        else:
            err = result.error.model_dump() if result.error else {"code": "UNKNOWN", "message": "", "retryable": False}
            if err.get("retryable") and attempt < s.max_retries:
                await stm.add_note(f"{name} attempt {attempt} failed ({err['code']}); retrying")
                update["error"] = None  # supervisor re-routes the same node
            elif name in FATAL_ON_ERROR or err.get("retryable"):
                update.update(error=err, fatal=True)
            else:
                update["error"] = err  # business failure -> supervisor routes to rejection
        return update

    node.__name__ = name
    return node


def make_supervisor_node(brain: SupervisorBrain):
    async def supervisor(state: LoanState) -> dict[str, Any]:
        s = get_settings()
        run_id_var.set(state["run_id"])
        steps = state.get("steps", 0) + 1
        if steps > s.supervisor_max_steps:
            err = {"code": "STEP_BUDGET_EXCEEDED", "message": f"exceeded {s.supervisor_max_steps} supervisor steps", "retryable": False}
            return {"next_node": "END", "fatal": True, "error": err, "steps": steps,
                    "audit": [_audit("supervisor", "aborted", err["message"])]}
        with tracing.span("supervisor:route", input={"step": steps}) as sp:
            decision = await brain.decide(state)
            sp.update(output=decision.model_dump(mode="json"))
        stm = ShortTermMemory(state["run_id"])
        await stm.save_status({"status": "running", "current_node": decision.next_node,
                               "trajectory": state.get("trajectory", [])})
        return {
            "next_node": decision.next_node,
            "instruction": decision.instruction,
            "steps": steps,
            "review_reasons": review_triggers(state),
            "routing_log": [{"step": steps, **decision.model_dump(mode="json")}],
            "audit": [_audit("supervisor", "routed", f"-> {decision.next_node} [{decision.source}, conf={decision.confidence}]")],
        }

    return supervisor


async def human_review(state: LoanState) -> dict[str, Any]:
    """HITL gate. `interrupt()` checkpoints the run and waits for an underwriter."""
    r = state.get("results", {})
    packet = {
        "run_id": state["run_id"], "applicant_id": state["applicant_id"], "reasons": review_triggers(state),
        "summary": {
            "loan": (r.get("validate_document") or {}).get("extracted"),
            "issues": (r.get("validate_document") or {}).get("issues"),
            "credit_score": (r.get("check_credit") or {}).get("credit_score"),
            "risk": {k: (r.get("assess_risk") or {}).get(k) for k in ("risk_grade", "risk_score", "dti", "ltv",
                                                                        "recommended_decision", "factors")},
            "compliance": {k: (r.get("check_compliance") or {}).get(k) for k in ("compliant", "pep", "aml_risk")},
        },
    }
    answer = interrupt(packet)  # <-- run pauses here (state persisted by the checkpointer)
    reviewer = str(answer.get("reviewer", "unknown"))
    try:
        hr = HumanReviewDecision.model_validate({k: v for k, v in answer.items() if k != "reviewer"})
    except Exception as exc:  # noqa: BLE001
        raise ValidationFailure(f"invalid human review payload: {exc}") from exc
    record = {"reviewer": reviewer, "decision": hr.decision, "notes": hr.reviewer_notes, "conditions": hr.conditions}
    return {"human_review": record, "trajectory": ["human_review"],
            "audit": [_audit(f"human:{reviewer}", "human_review", f"{hr.decision}: {hr.reviewer_notes}")]}


def build_graph(checkpointer=None, transport: AgentTransport | None = None):
    transport = transport or get_transport()
    brain = SupervisorBrain()
    g = StateGraph(LoanState)
    g.add_node("supervisor", make_supervisor_node(brain))
    for name in AGENT_NODES:
        g.add_node(name, make_agent_node(name, transport))
    g.add_node("human_review", human_review)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next_node"],
                            {**{n: n for n in AGENT_NODES}, "human_review": "human_review", "END": END})
    for name in (*AGENT_NODES, "human_review"):
        g.add_edge(name, "supervisor")
    return g.compile(checkpointer=checkpointer)
