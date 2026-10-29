"""ReAct sub-agent runtime (LangGraph StateGraph).

    START -> reason --(tool_calls)--> act --> reason ... --(final answer)--> END
                 \\--(max iterations)--> finalize --> END
    act --(high-risk call needing approval)--> END  (status = needs_approval, ProposedAction)

* reason  : LLM produces a Thought and either a tool call (Action) or the final answer.
* act     : executes tool calls through the allow-listed MCP session; observations are verified
            (pydantic), scanned for indirect prompt injection, defanged and fed back.
* HITL    : if a call needs human approval (shared policy), the agent does NOT call the tool; it
            returns a ProposedAction. After approval the supervisor re-invokes the agent in
            ``execute_approved`` mode with an approval token bound to the exact arguments.
Every node transition is checkpointed (checkpointer injectable: in-memory / Redis / Postgres).
"""
from __future__ import annotations

import json
import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from bankcore.agents.policies import approval_requirement, reviewer_summary
from bankcore.agents.specs import REACT_PROTOCOL, AgentSpec
from bankcore.config import Settings, get_settings
from bankcore.context.builder import ContextBuilder, ContextBundle
from bankcore.errors import BankAgentError, ToolNotPermitted
from bankcore.guardrails.injection import PromptInjectionDetector, neutralize_untrusted
from bankcore.guardrails.output_guard import guard_output
from bankcore.llm.factory import ResilientLLM
from bankcore.llm.structured import extract_json
from bankcore.logging import get_logger
from bankcore.mcp.client import MCPToolClient, MCPToolSession
from bankcore.observability.cost import UsageCallback
from bankcore.observability.metrics import AGENT_TASKS, GUARDRAIL_EVENTS, HITL_EVENTS
from bankcore.observability.tracing import langfuse_callbacks
from bankcore.schemas.common import AgentName, ErrorInfo
from bankcore.schemas.messages import (
    ComplianceVerdict, ComplianceViolation, ProposedAction, SubAgentRequest, SubAgentResponse, ToolCallRecord,
)
from bankcore.schemas.tools import TOOL_INPUT_MODELS
from bankcore.security.signing import issue_delegation_token  # noqa: F401  (re-export for tests)

log = get_logger(__name__)
_THOUGHT = re.compile(r"(?is)thought\s*:\s*(.+?)(?:\n|$)")

ANSWER_FORMAT = ("Final answer: plain text for the customer (max ~120 words), friendly and precise. "
                 "Lead with the outcome, then key figures, then any next step. Mask card numbers (last 4 only). "
                 "Do not mention tools, agents, JSON or internal reasoning.")
COMPLIANCE_FORMAT = ('Final answer: ONE JSON object {"compliant": bool, "violations": [{"rule": str, '
                     '"severity": "low|medium|high", "excerpt": str, "regulation": str}], "redacted_text": str, '
                     '"required_disclosures": [str]}')


class ReActState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    iterations: int
    records: Annotated[list[dict], operator.add]
    thoughts: Annotated[list[str], operator.add]
    observations: Annotated[list[dict], operator.add]
    proposed: dict | None
    final: str | None


def canonical_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    model = TOOL_INPUT_MODELS.get(tool)
    if model is None:
        return dict(args)
    return model.model_validate(args).model_dump(mode="json")


class ReActAgent:
    def __init__(self, spec: AgentSpec, llm: ResilientLLM, mcp: MCPToolClient, *, settings: Settings | None = None,
                 checkpointer=None) -> None:
        self.spec, self.llm, self.mcp = spec, llm, mcp
        self.settings = settings or get_settings()
        self.checkpointer = checkpointer or InMemorySaver()
        self.detector = PromptInjectionDetector()
        self.ctx_builder = ContextBuilder()

    # --------------------------------------------------------------- public --
    async def run(self, req: SubAgentRequest, *, trace_id: str) -> SubAgentResponse:
        usage_cb = UsageCallback(service=f"agent-{self.spec.name.value}", purpose="react")
        callbacks = [usage_cb, *langfuse_callbacks(trace_id)]
        try:
            async with self.mcp.session() as session:
                if req.mode == "execute_approved":
                    resp = await self._execute_approved(req, session, callbacks)
                else:
                    resp = await self._react(req, session, callbacks)
        except BankAgentError as e:
            log.warning("sub-agent failed", agent=self.spec.name.value, code=e.code, error=str(e))
            resp = SubAgentResponse(task_id=req.task_id, agent=self.spec.name, status="failed",
                                    answer=e.user_message,
                                    error=ErrorInfo(code=e.code, message=e.user_message, retryable=e.retryable))
        except Exception as e:  # noqa: BLE001
            log.exception("sub-agent crashed", agent=self.spec.name.value, error=str(e))
            resp = SubAgentResponse(task_id=req.task_id, agent=self.spec.name, status="failed",
                                    answer="I couldn't complete that step right now.",
                                    error=ErrorInfo(code="agent_internal_error", message="internal error",
                                                    retryable=True))
        resp.usage = usage_cb.usage
        AGENT_TASKS.labels(self.spec.name.value, resp.status).inc()
        return resp

    # ------------------------------------------------------------ react loop --
    def _initial_messages(self, req: SubAgentRequest, tools: list[dict]) -> list[BaseMessage]:
        is_compliance = self.spec.name == AgentName.COMPLIANCE
        facts = [f"[from {p.agent.value} agent] {p.summary}" for p in req.context.prior_results]
        facts += [f"Authenticated customer id: {req.user_id}"]
        bundle = ContextBundle(
            system_instructions=f"{self.spec.instructions}\n\n{REACT_PROTOCOL}",
            user_input=req.instruction,
            output_format=COMPLIANCE_FORMAT if is_compliance else ANSWER_FORMAT,
            tools=tools,
            retrieved_facts=facts,
            short_term_notes=req.context.short_term_notes,
            long_term_memory=[],
            max_tokens=self.settings.ollama_num_ctx - 1500,
        )
        msgs = self.ctx_builder.build(bundle)
        if req.context.customer_facts:
            msgs[1].content += "\n\nKnown customer preferences:\n" + "\n".join(
                f"- {neutralize_untrusted(f)}" for f in req.context.customer_facts)
        return msgs

    async def _react(self, req: SubAgentRequest, session: MCPToolSession, callbacks: list) -> SubAgentResponse:
        tools = await session.llm_tools()
        spec, s = self.spec, self.settings
        max_iter = min(spec.max_iterations, s.react_max_iterations + 2)

        async def reason(state: ReActState) -> dict:
            ai = await self.llm.ainvoke(state["messages"], tools=tools, config={"callbacks": callbacks},
                                        timeout_s=min(s.llm_timeout_s, max(2.0, req.deadline_ts - time.time())))
            thoughts = [m.strip() for m in _THOUGHT.findall(str(ai.content or ""))][:2]
            update: dict = {"messages": [ai], "iterations": state.get("iterations", 0) + 1, "thoughts": thoughts}
            if not ai.tool_calls:
                update["final"] = str(ai.content or "")
            return update

        async def act(state: ReActState) -> dict:
            ai: AIMessage = state["messages"][-1]
            out_msgs, records, observations = [], [], []
            for tc in ai.tool_calls:
                name, raw_args = tc["name"], tc.get("args") or {}
                call_id = tc.get("id") or name
                # --- state isolation: refuse tools outside the allow-list ---------------------
                if name not in spec.tools:
                    GUARDRAIL_EVENTS.labels("tool_call", "not_permitted").inc()
                    records.append(ToolCallRecord(tool=name, arguments={}, ok=False,
                                                  error_code="tool_not_permitted").model_dump())
                    out_msgs.append(ToolMessage(tool_call_id=call_id, content=json.dumps(
                        {"ok": False, "error": f"Tool '{name}' is not available to you."})))
                    continue
                # --- argument validation (lets the LLM self-correct) -------------------------
                try:
                    args = canonical_args(name, raw_args)
                except ValidationError as e:
                    msg = "; ".join(f"{'.'.join(map(str, er['loc']))}: {er['msg']}" for er in e.errors()[:4])
                    records.append(ToolCallRecord(tool=name, arguments=raw_args, ok=False,
                                                  error_code="invalid_arguments").model_dump())
                    out_msgs.append(ToolMessage(tool_call_id=call_id, content=json.dumps(
                        {"ok": False, "error": f"invalid arguments: {msg}"})))
                    continue
                # --- HITL: propose instead of executing ---------------------------------------
                req_ = approval_requirement(name, args, s)
                if spec.high_risk and req_.required:
                    action = ProposedAction(agent=spec.name, tool=name, arguments=args, risk_level=req_.risk,
                                            reason=req_.reason, summary_for_reviewer=reviewer_summary(name, args))
                    HITL_EVENTS.labels(spec.name.value, "proposed").inc()
                    out_msgs.append(ToolMessage(tool_call_id=call_id, content=json.dumps(
                        {"ok": False, "status": "pending_human_approval"})))
                    return {"messages": out_msgs, "records": records, "proposed": action.model_dump(mode="json")}
                env, latency = await session.call(name, args, delegation_token=req.delegation_token)
                if env.requires_approval:   # server-side enforcement caught something the agent missed
                    action = ProposedAction(agent=spec.name, tool=name, arguments=args, risk_level=req_.risk,
                                            reason="server requires approval",
                                            summary_for_reviewer=reviewer_summary(name, args))
                    return {"messages": out_msgs, "records": records, "proposed": action.model_dump(mode="json")}
                records.append(ToolCallRecord(tool=name, arguments=args, ok=env.ok, latency_ms=round(latency, 1),
                                              error_code=env.error.code if env.error else None).model_dump())
                payload = env.data if env.ok else {"error": env.error.model_dump() if env.error else "failed"}
                text = json.dumps({"ok": env.ok, **({"data": payload} if env.ok else payload)}, default=str)
                verdict = self.detector.scan(text, untrusted_source=True)
                if verdict.is_injection:
                    GUARDRAIL_EVENTS.labels("tool_output", "indirect_injection").inc()
                    text = json.dumps({"ok": env.ok, "warning": "content withheld: suspicious instructions detected",
                                       "data_summary": neutralize_untrusted(text)[:500]})
                observations.append({"tool": name, "ok": env.ok, "data": payload})
                out_msgs.append(ToolMessage(tool_call_id=call_id, content=f"<untrusted_data source=\"tool:{name}\">"
                                                                           f"{text}</untrusted_data>"))
            return {"messages": out_msgs, "records": records, "observations": observations}

        async def finalize(state: ReActState) -> dict:
            msgs = state["messages"] + [HumanMessage(content="Stop calling tools. Using only the observations above, "
                                                             "give your final answer now.")]
            ai = await self.llm.ainvoke(msgs, config={"callbacks": callbacks})
            return {"messages": [ai], "final": str(ai.content or "")}

        def after_reason(state: ReActState) -> str:
            if state.get("final") is not None:
                return END
            if state.get("iterations", 0) >= max_iter:
                return "finalize"
            return "act"

        def after_act(state: ReActState) -> str:
            return END if state.get("proposed") else "reason"

        g = StateGraph(ReActState)
        g.add_node("reason", reason)
        g.add_node("act", act)
        g.add_node("finalize", finalize)
        g.add_edge(START, "reason")
        g.add_conditional_edges("reason", after_reason, {"act": "act", "finalize": "finalize", END: END})
        g.add_conditional_edges("act", after_act, {"reason": "reason", END: END})
        g.add_edge("finalize", END)
        graph = g.compile(checkpointer=self.checkpointer)

        init = {"messages": self._initial_messages(req, tools), "iterations": 0, "records": [], "thoughts": [],
                "observations": [], "proposed": None, "final": None}
        cfg = {"configurable": {"thread_id": f"{req.thread_id}:{req.task_id}:{spec.name.value}"},
               "recursion_limit": max_iter * 2 + 6, "callbacks": callbacks}
        final_state = await graph.ainvoke(init, config=cfg)
        return self._to_response(req, final_state)

    # ----------------------------------------------------------- approved --
    async def _execute_approved(self, req: SubAgentRequest, session: MCPToolSession,
                                callbacks: list) -> SubAgentResponse:
        action = req.approved_action
        assert action is not None and req.approval_token
        if action.agent != self.spec.name or action.tool not in self.spec.tools:
            raise ToolNotPermitted("approved action does not belong to this agent")
        env, latency = await session.call(action.tool, action.arguments, delegation_token=req.delegation_token,
                                          approval_token=req.approval_token)
        record = ToolCallRecord(tool=action.tool, arguments=action.arguments, ok=env.ok, latency_ms=round(latency, 1),
                                error_code=env.error.code if env.error else None)
        HITL_EVENTS.labels(self.spec.name.value, "executed" if env.ok else "execution_failed").inc()
        if not env.ok:
            msg = env.error.message if env.error else "The approved action could not be completed."
            return SubAgentResponse(task_id=req.task_id, agent=self.spec.name, status="failed", answer=msg,
                                    tool_calls=[record], error=ErrorInfo(code=env.error.code if env.error else
                                                                         "failed", message=msg))
        answer = await self._phrase_result(action.tool, env.data or {}, callbacks)
        return SubAgentResponse(task_id=req.task_id, agent=self.spec.name, status="completed", answer=answer,
                                data={action.tool: env.data}, tool_calls=[record],
                                reasoning_trace=["Executed human-approved action"])

    async def _phrase_result(self, tool: str, data: dict, callbacks: list) -> str:
        """Confirmations of executed money/security actions use deterministic templates, not an LLM:
        zero hallucination risk on amounts and references, and no extra latency."""
        templates = {
            "initiate_internal_transfer": "Approved and completed: ${amount} was transferred from {source_acc} to "
                                          "{dest_acc}. Reference {transfer_id}. New source balance: "
                                          "${new_source_balance}.",
            "schedule_bill_pay": "Approved: ${amount} to {payee_name} is scheduled for {scheduled_date}. "
                                 "Reference {payment_id}.",
            "update_atm_withdrawal_limit": "Approved: the daily ATM limit on {card_id} changed from "
                                           "${previous_limit} to ${new_limit}.",
            "flag_fraudulent_transaction": "Approved: fraud case {case_id} is open for transaction {transaction_id} "
                                           "and is under investigation.",
            "lock_user_profile": "Approved: your online profile is locked for your protection and {sessions_revoked} "
                                 "active session(s) were signed out. Please call us to verify your identity and "
                                 "restore access.",
            "execute_fund_subscription": "Approved: your order {order_id} to invest ${amount} in {fund_id} was "
                                         "accepted and settles on {settlement_date}.",
        }
        try:
            return templates[tool].format(**data)
        except (KeyError, IndexError):
            ref = next((v for k, v in data.items() if k.endswith("_id")), "")
            return f"Your approved request has been completed successfully. Reference: {ref}."

    # ------------------------------------------------------------- result --
    def _to_response(self, req: SubAgentRequest, st: ReActState) -> SubAgentResponse:
        records = [ToolCallRecord.model_validate(r) for r in st.get("records", [])]
        data = {o["tool"]: o["data"] for o in st.get("observations", []) if o.get("ok")}
        if st.get("proposed"):
            action = ProposedAction.model_validate(st["proposed"])
            return SubAgentResponse(task_id=req.task_id, agent=self.spec.name, status="needs_approval",
                                    answer=f"This request needs approval from a bank officer: {action.reason}.",
                                    data=data, tool_calls=records, proposed_action=action,
                                    reasoning_trace=st.get("thoughts", []))
        final = st.get("final") or ""
        if self.spec.name == AgentName.COMPLIANCE:
            verdict = self._compliance_verdict(final, st.get("observations", []))
            return SubAgentResponse(task_id=req.task_id, agent=self.spec.name, status="completed",
                                    answer=verdict.redacted_text[:6000], data={"verdict": verdict.model_dump()},
                                    tool_calls=records, reasoning_trace=st.get("thoughts", []))
        evidence = [json.dumps(o.get("data"), default=str) for o in st.get("observations", [])]
        evidence += [p.summary for p in req.context.prior_results]
        guarded = guard_output(final, evidence=evidence)
        if guarded.blocked:
            GUARDRAIL_EVENTS.labels("agent_output", "blocked").inc()
        if guarded.ungrounded_figures:
            GUARDRAIL_EVENTS.labels("agent_output", "ungrounded_figures").inc()
            data["_ungrounded_figures"] = guarded.ungrounded_figures
        return SubAgentResponse(task_id=req.task_id, agent=self.spec.name, status="completed",
                                answer=guarded.text[:6000], data=data, tool_calls=records,
                                reasoning_trace=st.get("thoughts", []))

    @staticmethod
    def _compliance_verdict(final: str, observations: list[dict]) -> ComplianceVerdict:
        by_tool = {o["tool"]: o["data"] for o in observations if o.get("ok")}
        audit, red = by_tool.get("audit_compliance_check"), by_tool.get("redact_pii")
        if audit and red:      # deterministic, tool-grounded verdict (preferred)
            return ComplianceVerdict(
                compliant=bool(audit["compliant"]), redacted_text=red["redacted_text"],
                required_disclosures=audit.get("required_disclosures", []),
                violations=[ComplianceViolation(rule=f["rule"], severity=f["severity"], excerpt=f["excerpt"][:300],
                                                regulation=f["regulation"]) for f in audit.get("findings", [])])
        try:
            return ComplianceVerdict.model_validate(extract_json(final))
        except Exception:  # noqa: BLE001 — fail closed
            return ComplianceVerdict(compliant=False, redacted_text="",
                                     violations=[ComplianceViolation(rule="review_incomplete", severity="high",
                                                                     regulation="internal")])
