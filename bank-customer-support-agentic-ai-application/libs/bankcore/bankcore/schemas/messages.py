"""Pydantic contracts for every message that crosses an agent boundary.

These are validated twice: when produced (sender) and when consumed (receiver,
inside ``bankcore.security.signing.open_envelope``) — "message verification at handoff".
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from bankcore.schemas.common import AgentName, ErrorInfo, RiskLevel, StrictModel, TokenUsage


def _uuid() -> str:
    return uuid.uuid4().hex


def canonical_json(data: Any) -> str:
    """Deterministic JSON used for hashing / signing (sorted keys, no whitespace)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


# =============================================================================
# Public API (customer <-> supervisor)
# =============================================================================
class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$")
    channel: Literal["web", "mobile", "ivr", "chat"] = "web"


class TaskSummary(StrictModel):
    task_id: str
    agent: AgentName
    status: str
    tools_used: list[str] = Field(default_factory=list)


class PendingApprovalView(StrictModel):
    action_id: str
    agent: AgentName
    tool: str
    summary: str
    risk_level: RiskLevel


class ChatResponse(StrictModel):
    status: Literal["completed", "pending_approval", "blocked", "needs_clarification", "error"]
    reply: str
    session_id: str
    thread_id: str
    trace_id: str
    tasks: list[TaskSummary] = Field(default_factory=list)
    pending_approval: PendingApprovalView | None = None
    cached_route: bool = False


class ApprovalDecision(StrictModel):
    approved: bool
    note: str = Field(default="", max_length=500)


# =============================================================================
# Supervisor (Chain-of-Thought planner) output
# =============================================================================
class PlannedTask(StrictModel):
    task_id: str = Field(pattern=r"^t[0-9]{1,2}$")
    agent: AgentName
    instruction: str = Field(min_length=3, max_length=1000)
    depends_on: list[str] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)  # 1 = most urgent (fraud / lost card)

    @field_validator("agent")
    @classmethod
    def _no_direct_compliance(cls, v: AgentName) -> AgentName:
        if v == AgentName.COMPLIANCE:
            raise ValueError("compliance agent is invoked by the supervisor itself, not planned")
        return v


class SupervisorPlan(StrictModel):
    """Structured output of the CoT planner. ``reasoning`` holds the step-by-step chain of
    thought; it is stored in traces for audit but never shown to the customer."""
    reasoning: list[str] = Field(min_length=1, max_length=12)
    intents: list[str] = Field(default_factory=list, max_length=6)
    tasks: list[PlannedTask] = Field(default_factory=list, max_length=6)
    needs_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=500)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_graph(self) -> "SupervisorPlan":
        ids = [t.task_id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task_id in plan")
        known = set(ids)
        for t in self.tasks:
            unknown = set(t.depends_on) - known
            if unknown:
                raise ValueError(f"task {t.task_id} depends on unknown tasks {unknown}")
            if t.task_id in t.depends_on:
                raise ValueError("task cannot depend on itself")
        # cycle detection (Kahn)
        indeg = {t.task_id: len(t.depends_on) for t in self.tasks}
        children: dict[str, list[str]] = {i: [] for i in ids}
        for t in self.tasks:
            for d in t.depends_on:
                children[d].append(t.task_id)
        queue = [i for i, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            n = queue.pop()
            seen += 1
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if seen != len(ids):
            raise ValueError("plan contains a dependency cycle")
        if not self.tasks and not self.needs_clarification:
            raise ValueError("plan must contain tasks or ask a clarification question")
        if self.needs_clarification and not self.clarification_question:
            raise ValueError("clarification_question required when needs_clarification")
        return self


# =============================================================================
# Supervisor <-> sub-agent handoff
# =============================================================================
class PriorResult(StrictModel):
    task_id: str
    agent: AgentName
    summary: str = Field(max_length=2000)


class HandoffContext(StrictModel):
    prior_results: list[PriorResult] = Field(default_factory=list)
    short_term_notes: list[str] = Field(default_factory=list, max_length=20)
    customer_facts: list[str] = Field(default_factory=list, max_length=20)
    original_user_message: str = Field(default="", max_length=4000)


class ProposedAction(StrictModel):
    """A high-risk tool call the sub-agent wants to make, paused for human approval."""
    action_id: str = Field(default_factory=_uuid)
    agent: AgentName
    tool: str
    arguments: dict[str, Any]
    risk_level: RiskLevel
    reason: str
    summary_for_reviewer: str
    args_sha256: str = ""

    @model_validator(mode="after")
    def _hash_args(self) -> "ProposedAction":
        digest = hashlib.sha256(canonical_json({"tool": self.tool, "arguments": self.arguments}).encode()).hexdigest()
        if self.args_sha256 and self.args_sha256 != digest:
            raise ValueError("proposed action arguments were tampered with")
        object.__setattr__(self, "args_sha256", digest)
        return self


class SubAgentRequest(StrictModel):
    task_id: str
    thread_id: str
    user_id: str
    instruction: str = Field(min_length=1, max_length=4000)
    context: HandoffContext = Field(default_factory=HandoffContext)
    mode: Literal["plan_and_execute", "execute_approved"] = "plan_and_execute"
    approved_action: ProposedAction | None = None
    approval_token: str | None = None
    delegation_token: str            # signed, short-lived "act on behalf of user X" token
    deadline_ts: float = Field(default_factory=lambda: time.time() + 25)

    @model_validator(mode="after")
    def _approved_mode(self) -> "SubAgentRequest":
        if self.mode == "execute_approved" and not (self.approved_action and self.approval_token):
            raise ValueError("execute_approved mode requires approved_action and approval_token")
        return self


class ToolCallRecord(StrictModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    latency_ms: float = 0.0
    error_code: str | None = None


class SubAgentResponse(StrictModel):
    task_id: str
    agent: AgentName
    status: Literal["completed", "needs_approval", "failed", "rejected"]
    answer: str = Field(default="", max_length=6000)
    data: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    proposed_action: ProposedAction | None = None
    reasoning_trace: list[str] = Field(default_factory=list)
    error: ErrorInfo | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)

    @model_validator(mode="after")
    def _consistency(self) -> "SubAgentResponse":
        if self.status == "needs_approval" and self.proposed_action is None:
            raise ValueError("needs_approval requires proposed_action")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed status requires error info")
        return self


class HandoffEnvelope(StrictModel):
    """Signed wrapper for every inter-agent message (integrity + authenticity + freshness)."""
    message_id: str = Field(default_factory=_uuid)
    trace_id: str
    thread_id: str
    sender: str
    recipient: str
    created_at: float = Field(default_factory=time.time)
    payload_type: str
    payload: dict[str, Any]
    payload_sha256: str
    signature: str


# =============================================================================
# Compliance agent verdict
# =============================================================================
class ComplianceViolation(StrictModel):
    rule: str
    severity: Literal["low", "medium", "high"]
    excerpt: str = Field(default="", max_length=300)
    regulation: str = ""


class ComplianceVerdict(StrictModel):
    compliant: bool
    violations: list[ComplianceViolation] = Field(default_factory=list)
    redacted_text: str
    required_disclosures: list[str] = Field(default_factory=list)


class AuditEvent(StrictModel):
    event_id: str = Field(default_factory=_uuid)
    event_type: str
    actor: str
    user_id: str | None = None
    thread_id: str | None = None
    trace_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
