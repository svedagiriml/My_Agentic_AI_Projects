"""Supervisor graph (LangGraph) — the 'brain' of the multi-agent system.

    START
      -> guard_input ──blocked──────────────────────────────────────────────┐
      -> load_memory                                                         │
      -> plan (CoT; semantic routing cache; keyword fallback) ──clarify────┤
      -> dispatch  ⟲ (one task per step: context switching between agents)  │
           │  └─needs approval─> human_approval (interrupt() => checkpoint, │
           │                     staff decides, resume) ─> dispatch          │
      -> synthesize (CoT composition of all task results)                    │
      -> compliance_review (silent Compliance agent; revise once if needed)  │
      -> finalize (output guard, memory write-back, audit events) <──────────┘
    END

State is checkpointed after every node (AsyncPostgresSaver in prod), so a workflow can wait
hours for a human approval, survive container restarts / scale-to-zero, and resume exactly
where it stopped. Nothing secret (tokens) is stored in state; PII is redacted before entry.
"""
from __future__ import annotations

import asyncio
import json
import operator
import time
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from bankcore.agents.specs import PROMPT_VERSION, SPECS
from bankcore.cache.semantic_cache import SemanticRoutingCache
from bankcore.config import Settings
from bankcore.context.builder import ContextBuilder, ContextBundle, json_schema_format
from bankcore.errors import GuardrailViolation, LLMError, LLMOutputParseError
from bankcore.guardrails import InputGuard
from bankcore.guardrails.output_guard import guard_output
from bankcore.llm.factory import ResilientLLM
from bankcore.llm.structured import invoke_structured
from bankcore.logging import get_logger
from bankcore.memory.manager import MemoryManager
from bankcore.messaging.event_bus import TOPIC_AUDIT, TOPIC_HITL_DECIDED, TOPIC_HITL_REQUESTED, TOPIC_MEMORY, EventBus
from bankcore.observability import tracing
from bankcore.observability.metrics import GUARDRAIL_EVENTS, HITL_EVENTS
from bankcore.schemas.common import AgentName, ErrorInfo
from bankcore.schemas.memory import ConversationTurn, RetrievedMemory
from bankcore.schemas.messages import (
    HandoffContext, PlannedTask, PriorResult, ProposedAction, SubAgentRequest, SubAgentResponse, SupervisorPlan,
)
from bankcore.security.signing import issue_approval_token, issue_delegation_token

from supervisor.app.agent_client import SubAgentClient
from supervisor.app.fallback_router import keyword_plan
from supervisor.app.hitl import HitlRegistry
from supervisor.app.prompts import (
    PLANNER_FEWSHOT, PLANNER_INSTRUCTIONS, REVISE_INSTRUCTIONS, SYNTH_FORMAT, SYNTH_INSTRUCTIONS,
)

log = get_logger(__name__)


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class SupervisorState(TypedDict, total=False):
    user_id: str
    session_id: str
    thread_id: str
    trace_id: str
    channel: str
    message: str                       # PII-redacted customer text
    deadline_ts: float
    status: str
    guard: dict
    memory: dict
    plan: dict | None
    plan_source: str
    results: Annotated[dict, _merge]   # task_id -> SubAgentResponse (json)
    pending: dict | None               # {"task_id", "action"}
    hitl_decisions: Annotated[list, operator.add]
    draft: str
    compliance: dict
    final: str
    trajectory: Annotated[list, operator.add]


@dataclass
class SupervisorDeps:
    settings: Settings
    llm: ResilientLLM | None
    agents: SubAgentClient
    memory: MemoryManager
    bus: EventBus
    hitl: HitlRegistry
    cache: SemanticRoutingCache | None = None
    input_guard: InputGuard | None = None
    parallel_independent: bool = False
    compliance_mode: str = "always"    # always | risk_based


_AGENT_TOOLS = [{"type": "function", "function": {"name": f"delegate_to_{s.name.value}",
                                                  "description": s.routing_description}}
                for s in SPECS.values() if s.name != AgentName.COMPLIANCE]
_RISKY = {AgentName.PAYMENTS, AgentName.FRAUD, AgentName.LOANS, AgentName.INVESTMENTS, AgentName.CARDS}


def _results(state: SupervisorState) -> dict[str, SubAgentResponse]:
    return {k: SubAgentResponse.model_validate(v) for k, v in (state.get("results") or {}).items()}


def build_supervisor_graph(deps: SupervisorDeps, checkpointer):
    s = deps.settings
    ctx_builder = ContextBuilder()
    guard = deps.input_guard or InputGuard()

    # ------------------------------------------------------------ 1. guard --
    async def guard_input(state: SupervisorState) -> dict:
        res = guard.check(state["message"])
        info = {"allowed": res.allowed, "score": res.injection_score, "signals": res.signals, "reason": res.reason}
        tracing.score(state["trace_id"], "input_injection_score", res.injection_score)
        if not res.allowed:
            GUARDRAIL_EVENTS.labels("input", res.reason).inc()
            await deps.bus.safe_publish(TOPIC_AUDIT, {"event_type": "guardrail_block", "user_id": state["user_id"],
                                                      "trace_id": state["trace_id"], "details": info})
            return {"status": "blocked", "guard": info, "final": GuardrailViolation.user_message,
                    "trajectory": [f"guard_input:blocked:{res.reason}"]}
        return {"guard": info, "message": res.sanitized_text, "status": "running", "trajectory": ["guard_input:ok"]}

    # ----------------------------------------------------------- 2. memory --
    async def load_memory(state: SupervisorState) -> dict:
        mc = await deps.memory.load(user_id=state["user_id"], session_id=state["session_id"], query=state["message"])
        return {"memory": {
            "history": [t.model_dump() for t in mc.history],
            "notes": mc.notes,
            "long_term": [m.model_dump(mode="json") for m in (mc.semantic + mc.episodic)],
            "procedural": [m.content for m in mc.procedural],
        }, "trajectory": ["load_memory"]}

    # ------------------------------------------------------------- 3. plan --
    async def plan(state: SupervisorState, config: RunnableConfig) -> dict:
        msg, mem = state["message"], state.get("memory") or {}
        source, plan_obj = "llm", None
        if deps.cache is not None:
            hit = await deps.cache.lookup(msg)
            if hit:
                plan_obj, source = hit[0], "cache"
        if plan_obj is None:
            if deps.llm is None:
                plan_obj, source = keyword_plan(msg), "fallback"
            else:
                bundle = ContextBundle(
                    system_instructions=PLANNER_INSTRUCTIONS + "\n\n" + PLANNER_FEWSHOT,
                    user_input=msg,
                    output_format=json_schema_format(SupervisorPlan),
                    tools=_AGENT_TOOLS,
                    retrieved_facts=[f"Similar request solved before: {p}" for p in mem.get("procedural", [])],
                    short_term_notes=mem.get("notes", []),
                    history=[ConversationTurn.model_validate(t) for t in mem.get("history", [])],
                    long_term_memory=[RetrievedMemory.model_validate(m) for m in mem.get("long_term", [])],
                )
                try:
                    plan_obj = await invoke_structured(deps.llm, ctx_builder.build(bundle), SupervisorPlan,
                                                       retries=1, config=config)
                except (LLMError, LLMOutputParseError) as e:
                    log.warning("planner degraded to keyword routing", error=str(e)[:200])
                    plan_obj, source = keyword_plan(msg), "fallback"
                if source == "llm" and deps.cache is not None:
                    await deps.cache.store(msg, plan_obj)
        # Security-first ordering is enforced in code, not left to the model.
        if any(t.agent == AgentName.FRAUD for t in plan_obj.tasks):
            ordered = sorted(plan_obj.tasks, key=lambda t: 0 if t.agent == AgentName.FRAUD else 1)
            plan_obj = plan_obj.model_copy(update={"tasks": ordered})
        traj = [f"plan:{source}:" + ",".join(t.agent.value for t in plan_obj.tasks)]
        if plan_obj.needs_clarification and not plan_obj.tasks:
            return {"plan": plan_obj.model_dump(mode="json"), "plan_source": source, "status": "needs_clarification",
                    "final": plan_obj.clarification_question or "Could you tell me a bit more?", "trajectory": traj}
        return {"plan": plan_obj.model_dump(mode="json"), "plan_source": source, "trajectory": traj}

    # --------------------------------------------------------- 4. dispatch --
    def _request(state: SupervisorState, task: PlannedTask, done: dict[str, SubAgentResponse]) -> SubAgentRequest:
        mem = state.get("memory") or {}
        prior = [PriorResult(task_id=k, agent=v.agent, summary=v.answer[:1500]) for k, v in done.items()
                 if v.answer]
        facts = [m["content"] for m in mem.get("long_term", []) if m.get("kind") == "semantic"][:5]
        return SubAgentRequest(
            task_id=task.task_id, thread_id=state["thread_id"], user_id=state["user_id"], instruction=task.instruction,
            context=HandoffContext(prior_results=prior, short_term_notes=mem.get("notes", [])[-8:],
                                   customer_facts=facts, original_user_message=state["message"]),
            delegation_token=issue_delegation_token(user_id=state["user_id"], agent=task.agent.value,
                                                    thread_id=state["thread_id"], settings=s),
            deadline_ts=min(state["deadline_ts"], time.time() + s.subagent_timeout_s))

    async def dispatch(state: SupervisorState) -> dict:
        p = SupervisorPlan.model_validate(state["plan"])
        done = _results(state)
        ready = [t for t in p.tasks if t.task_id not in done and all(d in done for d in t.depends_on)]
        if not ready:
            return {}
        batch = ready if deps.parallel_independent else ready[:1]   # default: sequential context switching
        responses = await asyncio.gather(*[deps.agents.invoke(t.agent, _request(state, t, done),
                                                              trace_id=state["trace_id"]) for t in batch])
        update: dict[str, Any] = {"results": {}, "trajectory": []}
        for t, r in zip(batch, responses):
            update["results"][t.task_id] = r.model_dump(mode="json")
            update["trajectory"] += [f"{t.agent.value}:{c.tool}" for c in r.tool_calls] or [f"{t.agent.value}:-"]
            if r.status == "needs_approval" and r.proposed_action and not update.get("pending"):
                a = r.proposed_action
                update["pending"] = {"task_id": t.task_id, "action": a.model_dump(mode="json")}
                update["trajectory"].append(f"{t.agent.value}:proposed:{a.tool}")
                view = {"user_id": state["user_id"], "session_id": state["session_id"], "task_id": t.task_id,
                        "agent": a.agent.value, "tool": a.tool, "summary": a.summary_for_reviewer,
                        "risk_level": a.risk_level.value, "reason": a.reason, "action_id": a.action_id,
                        "trace_id": state["trace_id"]}
                await deps.hitl.add(state["thread_id"], view)
                await deps.bus.safe_publish(TOPIC_HITL_REQUESTED, view)
                HITL_EVENTS.labels(a.agent.value, "requested").inc()
        return update

    def after_dispatch(state: SupervisorState) -> str:
        if state.get("pending"):
            return "human_approval"
        p = SupervisorPlan.model_validate(state["plan"])
        done = state.get("results") or {}
        return "dispatch" if any(t.task_id not in done for t in p.tasks) else "synthesize"

    # --------------------------------------------------- 5. human approval --
    async def human_approval(state: SupervisorState) -> dict:
        pending = state["pending"]
        action = ProposedAction.model_validate(pending["action"])
        # Pause: the checkpoint is persisted; the API returns "pending_approval" to the customer.
        decision = interrupt({"type": "approval_required", "task_id": pending["task_id"],
                              "action": action.model_dump(mode="json")})
        # ---- resumed by POST /v1/hitl/{thread_id}/decision ----
        await deps.hitl.remove(state["thread_id"])
        approved = bool(decision.get("approved"))
        reviewer = str(decision.get("reviewer_id", "unknown"))
        prev = SubAgentResponse.model_validate(state["results"][pending["task_id"]])
        if approved and reviewer != state["user_id"]:
            token = issue_approval_token(action, user_id=state["user_id"], reviewer_id=reviewer, settings=s)
            req = SubAgentRequest(
                task_id=pending["task_id"], thread_id=state["thread_id"], user_id=state["user_id"],
                instruction=f"Execute approved action {action.tool}", mode="execute_approved",
                approved_action=action, approval_token=token,
                delegation_token=issue_delegation_token(user_id=state["user_id"], agent=action.agent.value,
                                                        thread_id=state["thread_id"], settings=s),
                deadline_ts=time.time() + s.subagent_timeout_s)
            resp = await deps.agents.invoke(action.agent, req, trace_id=state["trace_id"])
            resp = resp.model_copy(update={"tool_calls": prev.tool_calls + resp.tool_calls})
            event = "approved"
        else:
            note = str(decision.get("note") or "")[:300]
            resp = SubAgentResponse(task_id=pending["task_id"], agent=action.agent, status="rejected",
                                    answer="A bank officer reviewed this request and could not approve it"
                                           + (f" ({note})" if note else "") + ". Please contact us for details.",
                                    tool_calls=prev.tool_calls,
                                    error=ErrorInfo(code="hitl_rejected", message="rejected by reviewer"))
            event = "rejected"
        HITL_EVENTS.labels(action.agent.value, event).inc()
        tracing.score(state["trace_id"], "hitl_approved", 1.0 if event == "approved" else 0.0, comment=reviewer)
        await deps.bus.safe_publish(TOPIC_HITL_DECIDED, {"thread_id": state["thread_id"], "action_id": action.action_id,
                                                         "decision": event, "reviewer_id": reviewer,
                                                         "user_id": state["user_id"], "tool": action.tool})
        return {"results": {pending["task_id"]: resp.model_dump(mode="json")}, "pending": None,
                "hitl_decisions": [{"action_id": action.action_id, "decision": event, "reviewer_id": reviewer}],
                "trajectory": [f"hitl:{event}:{action.tool}"] +
                              ([f"{action.agent.value}:{action.tool}"] if event == "approved" and resp.status ==
                               "completed" else [])}

    # ------------------------------------------------------- 6. synthesize --
    async def synthesize(state: SupervisorState, config: RunnableConfig) -> dict:
        p = SupervisorPlan.model_validate(state["plan"])
        res = _results(state)
        ordered = [(t, res[t.task_id]) for t in p.tasks if t.task_id in res]
        if len(ordered) == 1 or deps.llm is None:
            return {"draft": "\n\n".join(r.answer for _, r in ordered if r.answer), "trajectory": ["synthesize:direct"]}
        facts = [f"Request {i + 1} ({t.agent.value}, status={r.status}): {r.answer}" for i, (t, r) in enumerate(ordered)]
        bundle = ContextBundle(system_instructions=SYNTH_INSTRUCTIONS, user_input=state["message"],
                               output_format=SYNTH_FORMAT, retrieved_facts=facts,
                               short_term_notes=(state.get("memory") or {}).get("notes", []))
        try:
            ai = await deps.llm.ainvoke(ctx_builder.build(bundle), config=config)
            draft = str(ai.content).strip()
            if not draft:
                raise LLMError("empty synthesis")
            return {"draft": draft, "trajectory": ["synthesize:llm"]}
        except Exception as e:  # noqa: BLE001
            log.warning("synthesis degraded", error=str(e)[:200])
            return {"draft": "\n\n".join(r.answer for _, r in ordered if r.answer),
                    "trajectory": ["synthesize:fallback"]}

    # ------------------------------------------------ 7. compliance review --
    async def _review(state: SupervisorState, text: str, suffix: str) -> SubAgentResponse:
        req = SubAgentRequest(
            task_id=f"compliance-{suffix}", thread_id=state["thread_id"], user_id=state["user_id"],
            instruction=f"Review this draft customer reply:\n{text}",
            delegation_token=issue_delegation_token(user_id=state["user_id"], agent="compliance",
                                                    thread_id=state["thread_id"], settings=s),
            deadline_ts=min(state["deadline_ts"], time.time() + s.subagent_timeout_s))
        return await deps.agents.invoke(AgentName.COMPLIANCE, req, trace_id=state["trace_id"])

    async def compliance_review(state: SupervisorState, config: RunnableConfig) -> dict:
        draft = state.get("draft", "")
        agents_used = {SubAgentResponse.model_validate(v).agent for v in (state.get("results") or {}).values()}
        if deps.compliance_mode == "risk_based" and not (agents_used & _RISKY):
            return {"compliance": {"skipped": True}, "trajectory": ["compliance:skipped"]}
        resp = await _review(state, draft, "1")
        verdict = (resp.data or {}).get("verdict") if resp.status == "completed" else None
        traj = ["compliance:" + ",".join(c.tool for c in resp.tool_calls)]
        if verdict is None:
            # Fail-safe degraded mode: deterministic PII scrub + generic disclosure.
            GUARDRAIL_EVENTS.labels("compliance", "degraded").inc()
            return {"draft": guard_output(draft).text, "compliance": {"degraded": True}, "trajectory": traj}
        if not verdict["compliant"] and deps.llm is not None:
            GUARDRAIL_EVENTS.labels("compliance", "revision").inc()
            findings = [f"{v['rule']} ({v['regulation']}): '{v['excerpt']}'" for v in verdict["violations"]]
            try:
                ai = await deps.llm.ainvoke([SystemMessage(content=REVISE_INSTRUCTIONS),
                                             HumanMessage(content="Findings:\n- " + "\n- ".join(findings) +
                                                          f"\n\nDraft:\n{verdict['redacted_text'] or draft}")],
                                            config=config)
                resp2 = await _review(state, str(ai.content), "2")
                v2 = (resp2.data or {}).get("verdict") if resp2.status == "completed" else None
                if v2:
                    verdict = v2
                    traj.append("compliance:revised")
            except Exception as e:  # noqa: BLE001
                log.warning("compliance revision failed", error=str(e)[:200])
        if not verdict["compliant"]:
            GUARDRAIL_EVENTS.labels("compliance", "blocked").inc()
            text = ("I'm sorry, I can't provide that information in this channel. A banker can help you directly — "
                    "would you like me to arrange a call back?")
        else:
            text = verdict["redacted_text"] or draft
            disclosures = [d for d in verdict.get("required_disclosures", []) if d not in text]
            if disclosures:
                text += "\n\n" + " ".join(f"Note: {d}" for d in disclosures)
        tracing.score(state["trace_id"], "compliance_pass", 1.0 if verdict["compliant"] else 0.0)
        return {"draft": text, "compliance": verdict, "trajectory": traj}

    # --------------------------------------------------------- 8. finalize --
    async def finalize(state: SupervisorState) -> dict:
        status = state.get("status", "running")
        res = _results(state)
        text = state.get("final") if status in ("blocked", "needs_clarification") else state.get("draft", "")
        evidence = [json.dumps(r.data, default=str) for r in res.values()] + [r.answer for r in res.values()]
        g = guard_output(text or "I'm sorry — I couldn't complete your request. Please try again.",
                         evidence=evidence if status == "running" else None)
        if g.blocked:
            GUARDRAIL_EVENTS.labels("output", "blocked").inc()
        if g.ungrounded_figures:
            GUARDRAIL_EVENTS.labels("output", "ungrounded_figures").inc()
            tracing.score(state["trace_id"], "ungrounded_figures", float(len(g.ungrounded_figures)))
        if status == "running":
            statuses = {r.status for r in res.values()}
            status = "error" if statuses and statuses <= {"failed"} else "completed"
        await deps.memory.record_turn(state["session_id"], state["message"], g.text, thread_id=state["thread_id"])
        try:
            agents = sorted({r.agent.value for r in res.values()})
            if agents:
                await deps.memory.st.add_note(state["session_id"], f"Handled via {', '.join(agents)}: "
                                                                  f"{state['message'][:120]}")
        except Exception:  # noqa: BLE001
            pass
        event = {"user_id": state["user_id"], "session_id": state["session_id"], "thread_id": state["thread_id"],
                 "trace_id": state["trace_id"], "message": state["message"], "reply": g.text, "status": status,
                 "plan": state.get("plan"), "plan_source": state.get("plan_source"),
                 "trajectory": state.get("trajectory", []), "task_statuses": {k: r.status for k, r in res.items()},
                 "hitl": state.get("hitl_decisions", []),
                 "compliance_ok": bool((state.get("compliance") or {}).get("compliant", True)),
                 "prompt_version": PROMPT_VERSION}
        await deps.bus.safe_publish(TOPIC_MEMORY, event)
        await deps.bus.safe_publish(TOPIC_AUDIT, {"event_type": "conversation_turn", **event})
        return {"final": g.text, "status": status, "trajectory": ["finalize"]}

    # ------------------------------------------------------------ wiring --
    g = StateGraph(SupervisorState)
    for name, fn in [("guard_input", guard_input), ("load_memory", load_memory), ("plan", plan),
                     ("dispatch", dispatch), ("human_approval", human_approval), ("synthesize", synthesize),
                     ("compliance_review", compliance_review), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", lambda st: "finalize" if st.get("status") == "blocked" else "load_memory",
                            {"finalize": "finalize", "load_memory": "load_memory"})
    g.add_edge("load_memory", "plan")
    g.add_conditional_edges("plan", lambda st: "finalize" if st.get("status") == "needs_clarification" else "dispatch",
                            {"finalize": "finalize", "dispatch": "dispatch"})
    g.add_conditional_edges("dispatch", after_dispatch,
                            {"human_approval": "human_approval", "dispatch": "dispatch", "synthesize": "synthesize"})
    g.add_conditional_edges("human_approval", after_dispatch,
                            {"human_approval": "human_approval", "dispatch": "dispatch", "synthesize": "synthesize"})
    g.add_edge("synthesize", "compliance_review")
    g.add_edge("compliance_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
