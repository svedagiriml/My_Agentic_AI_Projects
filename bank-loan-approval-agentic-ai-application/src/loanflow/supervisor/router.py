"""The supervisor's brain: Chain-of-Thought routing with a policy guard.

Order of precedence for each routing step
  1. Semantic cache  — has an equivalent situation been routed before?
  2. CoT LLM         — think step-by-step over the 7-part context, propose next node
  3. Rule policy     — deterministic CoT (also the fallback when no LLM is reachable)

Whatever proposes the route, `legal_next()` validates it. Legal moves are the policy
route plus *escalation to a human* (always the conservative direction). An LLM or a
poisoned cache entry can therefore never skip a check, approve without the gates
passing, or reject without a recorded reason.
"""
from __future__ import annotations

from typing import Any

from loanflow.common.config import get_settings
from loanflow.common.logging import get_logger
from loanflow.common.schemas import RoutingDecision
from loanflow.context.builder import ContextPack
from loanflow.llm.client import LLMUnavailable, get_llm
from loanflow.memory.long_term import get_long_term_memory
from loanflow.memory.semantic_cache import SemanticCache
from loanflow.observability.metrics import CACHE_EVENTS
from loanflow.supervisor.state import LoanState

log = get_logger(__name__)

SUPERVISOR_INSTRUCTIONS = """You are the Supervisor of a bank's multi-agent loan-processing system.
You coordinate specialist subagents; you never perform their work yourself.
Workflow: fetch_document -> validate_document -> check_credit -> assess_risk -> check_compliance
          -> (human_review when required) -> compile_report | compile_rejection -> END.
Reason with Chain-of-Thought: write your thoughts as numbered steps:
  (1) what has been completed, (2) what each result implies under credit policy,
  (3) whether any hard stop or review trigger applies, (4) which single node must run next.
Rules: a failed hard check routes to compile_rejection; review triggers route to human_review
before any compile step; after a compile step the run ENDs. When unsure, choose human_review."""

DECISION_NODES = ("compile_report", "compile_rejection")


def review_triggers(state: LoanState) -> list[str]:
    r = state.get("results", {})
    s = get_settings()
    reasons = []
    v = r.get("validate_document") or {}
    if v.get("requires_human_review"):
        reasons.append("document flagged by injection guardrail")
    ext = v.get("extracted") or {}
    if ext.get("loan_amount_inr", 0) > s.hitl_amount_threshold_inr:
        reasons.append(f"loan amount above INR {s.hitl_amount_threshold_inr:,.0f} threshold")
    if (r.get("assess_risk") or {}).get("recommended_decision") == "REFERRED":
        reasons.append(f"borderline risk grade {(r.get('assess_risk') or {}).get('risk_grade')}")
    if (r.get("check_compliance") or {}).get("requires_human_review"):
        reasons.append("compliance requires enhanced due diligence")
    return reasons


def situation(state: LoanState) -> str:
    """Canonical, PII-free, low-cardinality summary of where the run is (semantic-cache key)."""
    r = state.get("results", {})
    v, c, k, cp = (r.get(n) or {} for n in ("validate_document", "check_credit", "assess_risk", "check_compliance"))
    hr = state.get("human_review") or {}
    err = state.get("error") or {}
    parts = [
        "done=" + ",".join(n for n in ("fetch_document", "validate_document", "check_credit", "assess_risk",
                                        "check_compliance", "compile_report", "compile_rejection") if n in r),
        f"doc_valid={v.get('is_valid') if v else 'na'}",
        f"credit_ok={c.get('meets_minimum') if c else 'na'}",
        f"risk_rec={k.get('recommended_decision', 'na')}",
        f"compliant={cp.get('compliant') if cp else 'na'}",
        f"review_needed={bool(review_triggers(state))}",
        f"human={hr.get('decision', 'none')}",
        f"error={err.get('code', 'none')}:{'fatal' if state.get('fatal') else 'ok'}",
    ]
    return " | ".join(parts)


def rule_route(state: LoanState) -> RoutingDecision:
    """Deterministic policy expressed as an explicit chain of thought (auditable)."""
    r = state.get("results", {})
    t: list[str] = []
    done = [n for n in r]
    t.append(f"Completed so far: {done or 'nothing'}.")

    def d(node: str, thought: str, instruction: str = "", conf: float = 1.0) -> RoutingDecision:
        t.append(thought)
        t.append(f"Therefore the next node is {node}.")
        return RoutingDecision(thoughts=t, next_node=node, instruction=instruction, confidence=conf, source="rules")

    if any(n in r for n in DECISION_NODES):
        return d("END", "A final decision has been compiled and stored.")
    if state.get("fatal"):
        return d("END", f"Unrecoverable error {state.get('error', {}).get('code')}; a decision cannot be produced safely.")
    if err := state.get("error"):
        return d("compile_rejection", f"Business error {err.get('code')} from a check makes the application ineligible.",
                 "Compile a rejection citing the failed check.")
    if "fetch_document" not in r:
        return d("fetch_document", "No document has been retrieved; nothing can be assessed without it.",
                 "Fetch the requested document for the applicant.")
    if "validate_document" not in r:
        return d("validate_document", "The document is available but not validated.",
                 "Extract fields, check completeness and verify KYC.")
    v = r["validate_document"]
    if not v["is_valid"]:
        return d("compile_rejection", f"Validation failed: {v['issues']}. Policy: incomplete/KYC-failed applications are declined.",
                 "Compile a rejection citing the validation issues.")
    if "check_credit" not in r:
        return d("check_credit", "Document is valid; bureau history is the next gate.", "Confirm consent and pull the bureau report.")
    if not r["check_credit"]["meets_minimum"]:
        return d("compile_rejection", f"Credit score {r['check_credit']['credit_score']} or bureau history fails the minimum.",
                 "Compile a rejection citing bureau reasons.")
    if "assess_risk" not in r:
        return d("assess_risk", "Credit gate passed; affordability and risk must be scored.", "Score risk, DTI, LTV and pricing.")
    if r["assess_risk"]["recommended_decision"] == "REJECTED":
        return d("compile_rejection", f"Risk engine recommends rejection: {r['assess_risk']['factors']}.",
                 "Compile a rejection citing the hard stops.")
    if "check_compliance" not in r:
        return d("check_compliance", "Risk acceptable; regulatory and AML screening is mandatory before any approval.",
                 "Run sanctions, PEP, AML and RBI checks.")
    if not r["check_compliance"]["compliant"]:
        return d("compile_rejection", "Compliance failed (sanctions or regulatory breach) - must decline.",
                 "Compile a rejection; do not disclose list details to the customer.")
    triggers = review_triggers(state)
    hr = state.get("human_review")
    if triggers and not hr:
        return d("human_review", f"Review triggers present: {triggers}. A human underwriter must decide.")
    if hr:
        node = "compile_report" if hr["decision"] == "APPROVED" else "compile_rejection"
        return d(node, f"Underwriter decided {hr['decision']}.", f"Compile the {hr['decision'].lower()} decision.")
    return d("compile_report", "All gates passed with no review triggers.", "Compile the approval, store and notify.")


def legal_next(state: LoanState, policy: RoutingDecision) -> set[str]:
    legal = {policy.next_node}
    if policy.next_node not in ("END", "human_review") and not state.get("human_review") \
            and "fetch_document" in state.get("results", {}) and not state.get("fatal"):
        legal.add("human_review")  # escalation is always permitted
    return legal


class SupervisorBrain:
    def __init__(self) -> None:
        self.cache = SemanticCache("routing")

    async def decide(self, state: LoanState) -> RoutingDecision:
        policy = rule_route(state)
        legal = legal_next(state, policy)
        sit = situation(state)

        cached = await self.cache.get(sit)
        if cached:
            value, score = cached
            try:
                dec = RoutingDecision.model_validate({**value, "source": "cache"})
                if dec.next_node in legal:
                    CACHE_EVENTS.labels("semantic_routing", "hit").inc()
                    dec.thoughts = [f"Semantic cache hit (similarity {score}) for situation: {sit}", *dec.thoughts][:12]
                    # instructions are run-agnostic, but refresh from policy to stay current
                    return dec.model_copy(update={"instruction": policy.instruction or dec.instruction})
            except Exception:  # noqa: BLE001 - corrupt entry: ignore
                pass
            CACHE_EVENTS.labels("semantic_routing", "rejected").inc()
        CACHE_EVENTS.labels("semantic_routing", "miss").inc()

        decision = policy
        try:
            decision = await self._llm_decide(state, sit, policy)
            if decision.next_node not in legal:
                log.warning("LLM route rejected by policy guard",
                            extra={"extra_fields": {"proposed": decision.next_node, "legal": sorted(legal)}})
                decision = policy.model_copy(update={"thoughts": [*policy.thoughts,
                                                                  f"(LLM proposed illegal route {decision.next_node}; overridden)"][:12]})
        except LLMUnavailable:
            decision = policy
        except Exception as exc:  # noqa: BLE001
            log.warning("supervisor LLM failed; using policy", extra={"extra_fields": {"error": str(exc)[:200]}})
            decision = policy
        if decision.next_node == policy.next_node:  # only cache policy-consistent routes
            await self.cache.put(sit, decision.model_dump(mode="json", exclude={"source"}))
        return decision

    async def _llm_decide(self, state: LoanState, sit: str, policy: RoutingDecision) -> RoutingDecision:
        memories = await get_long_term_memory().recall(
            "routing policy human review rejection " + sit, kinds=["procedural", "semantic"],
            namespaces=["global"], k=4)
        compact = {n: _compact(o) for n, o in state.get("results", {}).items()}
        pack = ContextPack(
            system_instructions=SUPERVISOR_INSTRUCTIONS,
            tools=[{"name": n, "description": f"subagent {n}"} for n in
                   ("fetch_document", "validate_document", "check_credit", "assess_risk", "check_compliance",
                    "human_review", "compile_report", "compile_rejection", "END")],
            output_format=RoutingDecision.model_json_schema(),
            user_input=f"Decide the next step for loan run {state['run_id']} "
                       f"(applicant {state['applicant_id']}, document {state['document_id']}).",
            retrieved_facts={"situation": sit, "results": compact, "review_triggers": review_triggers(state),
                             "error": state.get("error"), "human_review": state.get("human_review")},
            short_term_notes=state.get("notes", []),
            long_term_memory=memories,
        )
        system, user = pack.render()
        res = await get_llm().chat_json(system=system, user=user, schema=RoutingDecision, name="supervisor.cot")
        return res.parsed.model_copy(update={"source": "llm"})  # type: ignore[union-attr]


def _compact(out: Any) -> Any:
    if not isinstance(out, dict):
        return out
    return {k: v for k, v in out.items() if k not in ("content", "audit_trail")}
