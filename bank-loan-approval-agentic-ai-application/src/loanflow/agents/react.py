"""ReAct (Reason + Act) engine shared by all subagents.

Loop:  Thought -> Action(tool, args) -> Observation -> ... -> Finish
  * The LLM proposes each step as JSON (`ReActStepOut`), built with the 7-part context.
  * Guards around the LLM:
      - action must be in the agent's MCP allow-list (else observation = error)
      - `finish` is refused until every *required* tool has succeeded
        (an LLM cannot skip the credit pull and invent a score)
      - after 2 invalid proposals, or if no LLM is reachable, the agent's
        deterministic planner takes over (same Thought/Action/Observation trace)
  * The final typed output is assembled by the agent's `finalize()` from *tool
    observations*, never from free LLM text -> numbers are grounded by construction.
    The LLM may contribute narrative (`final_notes`), which is grounding-checked.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from loanflow.common.config import get_settings
from loanflow.common.exceptions import AgentExecutionError, LoanFlowError
from loanflow.common.logging import get_logger, node_var
from loanflow.common.schemas import ReActStep, RunMetrics, Strict, ToolCallRecord
from loanflow.context.builder import ContextPack
from loanflow.llm.client import LLMUnavailable, get_llm
from loanflow.mcp_layer.client import MCPToolbox
from loanflow.memory.long_term import MemoryHit, get_long_term_memory
from loanflow.observability import tracing

log = get_logger(__name__)

REACT_INSTRUCTIONS = """You are operating in the ReAct pattern.
At each step output JSON with:
  "thought": your reasoning about what is known and what is missing,
  "action": the exact name of ONE tool from the TOOLS section, or "finish" when done,
  "action_input": the tool arguments (object) - {} for finish,
  "final_notes": (finish only) a short factual note for the audit trail.
Call each required tool before finishing. Never invent tool results or numbers."""


class BusinessToolError(LoanFlowError):
    """A tool answered with a business error (NOT_FOUND, FORBIDDEN, IDENTITY_MISMATCH...)."""

    def __init__(self, code: str, message: str):
        super().__init__(message, details={"tool_error_code": code})
        self.code = code
        self.http_status = 422


def raise_tool_errors(obs: dict[str, Any]) -> None:
    errs = obs.get("_errors") or {}
    if errs:
        tool, err = next(iter(errs.items()))
        err = err or {}
        raise BusinessToolError(str(err.get("code", "TOOL_ERROR")), f"{tool}: {err.get('message', 'failed')}")


class ReActStepOut(BaseModel):
    thought: str = Field(max_length=1500)
    action: str = Field(max_length=80)
    action_input: dict[str, Any] = Field(default_factory=dict)
    final_notes: str | None = Field(default=None, max_length=1500)


class PlannedAction(BaseModel):
    thought: str
    tool: str | None  # None => finish
    args: dict[str, Any] = Field(default_factory=dict)


class AgentRunOutput(BaseModel):
    output: dict[str, Any]
    trace: list[ReActStep]
    metrics: RunMetrics
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    needs_review: bool = False


class BaseReActAgent(ABC):
    name: str
    description: str
    role_prompt: str
    tools: dict[str, list[str]]  # server -> allowed tool names
    required_tools: tuple[str, ...] = ()
    output_model: type[Strict]
    memory_kinds: tuple[str, ...] = ("semantic", "procedural")

    # ---- hooks -------------------------------------------------------
    def prepare(self, task: dict[str, Any]) -> dict[str, Any]:
        """Optional pre-computation (e.g. drafting the decision before rendering it)."""
        return {}

    @abstractmethod
    def plan(self, task: dict[str, Any], obs: dict[str, Any], scratch: dict[str, Any]) -> PlannedAction:
        """Deterministic policy: next action given observations so far."""

    @abstractmethod
    def finalize(self, task: dict[str, Any], obs: dict[str, Any], scratch: dict[str, Any], notes: str | None) -> dict[str, Any]:
        """Assemble the typed output from tool observations."""

    @staticmethod
    def next_in_sequence(sequence: list[tuple[str, str, dict[str, Any]]], obs: dict[str, Any]) -> PlannedAction:
        """Helper for deterministic planners: first (tool, thought, args) not yet observed; finish on any error."""
        if obs.get("_errors"):
            return PlannedAction(thought=f"A tool returned an error {list(obs['_errors'])}; stop and report.", tool=None)
        for tool, thought, args in sequence:
            if tool not in obs:
                return PlannedAction(thought=thought, tool=tool, args=args)
        return PlannedAction(thought="All required information gathered; assembling output.", tool=None)

    def memory_query(self, task: dict[str, Any]) -> str:
        return f"{self.name} {self.description}"

    def needs_review(self, output: dict[str, Any]) -> bool:
        return bool(output.get("requires_human_review"))

    # ---- engine ------------------------------------------------------
    async def run(self, task: dict[str, Any]) -> AgentRunOutput:
        s = get_settings()
        node_var.set(self.name)
        t0 = time.perf_counter()
        metrics = RunMetrics()
        trace: list[ReActStep] = []
        obs: dict[str, Any] = {}  # tool name -> last successful payload
        scratch = self.prepare(task)
        memories = await self._recall(task)

        tb = MCPToolbox(self.tools)
        try:
            return await self._run(task, s, t0, metrics, trace, obs, scratch, memories, tb)
        except LoanFlowError as exc:
            exc.trace = trace  # type: ignore[attr-defined]  # keep partial trajectory for audit/eval
            exc.tool_records = list(tb.records)  # type: ignore[attr-defined]
            raise

    async def _run(self, task, s, t0, metrics, trace, obs, scratch, memories, tb) -> AgentRunOutput:
        use_llm = True
        invalid = 0
        final_notes: str | None = None
        with tracing.span(f"agent:{self.name}", input={"task": task.get("instruction")}) as sp:
            async with tb:
                catalog = tb.catalog
                tool_names = {t["name"] for t in catalog}
                for step in range(1, s.react_max_steps + 1):
                    proposal: PlannedAction | None = None
                    if use_llm:
                        try:
                            proposal, notes = await self._llm_step(task, catalog, obs, trace, memories, metrics)
                            final_notes = notes or final_notes
                        except LLMUnavailable:
                            use_llm = False
                    if proposal is not None and proposal.tool is not None and proposal.tool not in tool_names:
                        trace.append(ReActStep(step=step, thought=proposal.thought, action=proposal.tool,
                                               action_input=proposal.args, observation="ERROR: unknown or disallowed tool"))
                        invalid += 1
                        if invalid >= 2:
                            use_llm = False
                        continue
                    if proposal is not None and proposal.tool is None:
                        missing = [] if obs.get("_errors") else [t for t in self.required_tools if t not in obs]
                        if missing:
                            trace.append(ReActStep(step=step, thought=proposal.thought, action="finish",
                                                   observation=f"REFUSED: required tools not yet called: {missing}"))
                            invalid += 1
                            if invalid >= 2:
                                use_llm = False
                            continue
                    if proposal is None:
                        proposal = self.plan(task, obs, scratch)
                    if proposal.tool is None:
                        trace.append(ReActStep(step=step, thought=proposal.thought, action="finish"))
                        break
                    observation = await self._act(tb, proposal, obs, metrics)
                    trace.append(ReActStep(step=step, thought=proposal.thought, action=proposal.tool,
                                           action_input=proposal.args, observation=observation[:800]))
                else:
                    missing = [] if obs.get("_errors") else [t for t in self.required_tools if t not in obs]
                    if missing:
                        raise AgentExecutionError(f"{self.name}: step budget exhausted", details={"missing": missing})
                records = list(tb.records)

            try:
                output = self.finalize(task, obs, scratch, final_notes)
                validated = self.output_model.model_validate(output).model_dump(mode="json")
            except ValidationError as exc:
                raise AgentExecutionError(f"{self.name} produced invalid output", details={"errors": exc.errors()[:5]}) from exc
            metrics.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            metrics.tool_calls = len(records)
            sp.update(output=validated)
        return AgentRunOutput(output=validated, trace=trace, metrics=metrics, tool_calls=records,
                              needs_review=self.needs_review(validated))

    async def _recall(self, task: dict[str, Any]) -> list[MemoryHit]:
        try:
            ns = ["global", f"applicant:{task['applicant_id']}"]
            return await get_long_term_memory().recall(self.memory_query(task), kinds=list(self.memory_kinds),
                                                       namespaces=ns, k=4)
        except Exception as exc:  # noqa: BLE001 - memory is best-effort
            log.warning("memory recall failed", extra={"extra_fields": {"error": str(exc)}})
            return []

    async def _llm_step(self, task, catalog, obs, trace, memories, metrics) -> tuple[PlannedAction, str | None]:
        pack = ContextPack(
            system_instructions=f"{self.role_prompt}\n\n{REACT_INSTRUCTIONS}",
            tools=[{k: v for k, v in t.items() if k != "server"} for t in catalog],
            output_format=ReActStepOut.model_json_schema(),
            user_input=f"Applicant {task['applicant_id']}, document {task['document_id']}. Directive: {task.get('instruction', '')}",
            retrieved_facts={"upstream": task.get("context", {}), "observations": obs,
                             "previous_steps": [t.model_dump() for t in trace[-4:]]},
            short_term_notes=task.get("notes", []),
            long_term_memory=memories,
        )
        system, user = pack.render()
        res = await get_llm().chat_json(system=system, user=user, schema=ReActStepOut, name=f"{self.name}.react", small=True)
        metrics.llm_calls += 1
        metrics.prompt_tokens += res.usage.prompt_tokens
        metrics.completion_tokens += res.usage.completion_tokens
        metrics.cost_usd += res.usage.cost_usd
        metrics.cache_hits += int(res.usage.cached)
        out: ReActStepOut = res.parsed  # type: ignore[assignment]
        tool = None if out.action.strip().lower() == "finish" else out.action.strip()
        return PlannedAction(thought=out.thought, tool=tool, args=out.action_input), out.final_notes

    async def _act(self, tb: MCPToolbox, p: PlannedAction, obs: dict[str, Any], metrics: RunMetrics) -> str:
        try:
            payload = await tb.call(p.tool, p.args)  # type: ignore[arg-type]
        except LoanFlowError as exc:
            if exc.retryable:  # infra failure (timeout / circuit open): fail fast, supervisor retries
                raise
            obs.setdefault("_errors", {})[p.tool] = {"code": exc.code, "message": exc.message[:300]}
            return f"ERROR: {exc.code}: {exc.message[:300]}"  # e.g. bad args -> LLM can self-correct
        if payload.get("ok", True):
            obs[p.tool] = payload
            errs = obs.get("_errors") or {}
            errs.pop(p.tool, None)  # a corrected retry clears the earlier error
            if not errs:
                obs.pop("_errors", None)
        else:
            obs.setdefault("_errors", {})[p.tool] = payload.get("error")
        return json.dumps(payload, default=str)
