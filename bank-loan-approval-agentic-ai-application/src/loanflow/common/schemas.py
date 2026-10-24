"""Pydantic data contracts — every JSON message in the system is one of these.

Two layers:
  * Domain payloads  – the typed output of each subagent (what the work produced)
  * Envelopes        – AgentTask / AgentResult wrap payloads for handoff; they carry
                       ids, deadlines, a SHA-256 checksum and an HMAC signature that is
                       verified at every handoff (see common/integrity.py).

`extra="forbid"` everywhere: an LLM or a compromised service cannot smuggle extra
fields (e.g. "override_decision": true) through a handoff.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"
POLICY_VERSION = "credit-policy-2026.10"


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


# ---------------------------------------------------------------- enums
class Decision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REFERRED = "REFERRED"  # sent to a human underwriter


class RiskGrade(str, Enum):
    A = "A"  # very low risk
    B = "B"
    C = "C"  # borderline -> HITL
    D = "D"
    E = "E"  # decline


NodeName = Literal[
    "fetch_document",
    "validate_document",
    "check_credit",
    "assess_risk",
    "check_compliance",
    "human_review",
    "compile_report",
    "compile_rejection",
    "END",
]
AGENT_NODES: tuple[str, ...] = (
    "fetch_document",
    "validate_document",
    "check_credit",
    "assess_risk",
    "check_compliance",
    "compile_report",
    "compile_rejection",
)


# ---------------------------------------------------------------- API inputs
class LoanApplicationRequest(Strict):
    applicant_id: str = Field(pattern=r"^A\d{4,10}$", description="Applicant identifier, e.g. A1001")
    document_id: str = Field(pattern=r"^D-[A-Z0-9-]{3,40}$", description="Document identifier, e.g. D-1001-APP")
    idempotency_key: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    channel: Literal["branch", "web", "mobile", "partner"] = "web"
    notes: str | None = Field(default=None, max_length=500, description="Free-text from the loan officer (guardrailed)")


class HumanReviewDecision(Strict):
    decision: Literal["APPROVED", "REJECTED"]
    reviewer_notes: str = Field(min_length=5, max_length=2000)
    conditions: list[str] = Field(default_factory=list, max_length=10)


# ---------------------------------------------------------------- domain payloads
class FetchDocumentOutput(Strict):
    document_id: str
    applicant_id: str
    doc_type: Literal["loan_application", "salary_slip", "bank_statement", "itr", "unknown"]
    content: str = Field(description="PII-redacted document text")
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pii_redacted: list[str] = Field(default_factory=list, description="Types of PII that were redacted")
    injection_flags: list[str] = Field(default_factory=list, description="Prompt-injection indicators found in the doc")
    source_system: str = "dms"


class ExtractedApplication(Strict):
    applicant_name: str
    employment_type: Literal["salaried", "self_employed", "unemployed", "retired", "unknown"]
    monthly_income_inr: float = Field(ge=0)
    existing_emi_inr: float = Field(ge=0)
    loan_amount_inr: float = Field(gt=0)
    tenure_months: int = Field(ge=6, le=360)
    loan_type: Literal["Home Loan", "Personal Loan", "Business Loan", "Auto Loan"]
    loan_purpose: str
    property_value_inr: float | None = Field(default=None, ge=0)
    pan_last4: str | None = Field(default=None, pattern=r"^[A-Z0-9]{4}$")


class ValidateDocumentOutput(Strict):
    is_valid: bool
    extracted: ExtractedApplication | None = None
    missing_fields: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    kyc_verified: bool = False
    requires_human_review: bool = False

    @model_validator(mode="after")
    def _valid_needs_data(self) -> ValidateDocumentOutput:
        if self.is_valid and self.extracted is None:
            raise ValueError("is_valid=True requires extracted application data")
        return self


class CheckCreditOutput(Strict):
    bureau: Literal["CIBIL", "Experian", "Equifax", "CRIF"] = "CIBIL"
    credit_score: int = Field(ge=300, le=900)
    active_loans: int = Field(ge=0)
    defaults_last_24m: int = Field(ge=0)
    enquiries_last_6m: int = Field(ge=0)
    credit_utilisation_pct: float = Field(ge=0, le=100)
    flags: list[str] = Field(default_factory=list)
    meets_minimum: bool


class AssessRiskOutput(Strict):
    risk_grade: RiskGrade
    risk_score: float = Field(ge=0, le=1, description="Probability-of-default proxy")
    proposed_emi_inr: float = Field(ge=0)
    dti: float = Field(ge=0, description="(existing EMI + new EMI) / monthly income")
    ltv: float | None = Field(default=None, ge=0)
    factors: list[str]
    recommended_decision: Decision
    suggested_rate_pct: float = Field(ge=0, le=36)


class ComplianceCheck(Strict):
    name: str
    passed: bool
    detail: str
    regulation: str


class CheckComplianceOutput(Strict):
    compliant: bool
    sanctions_hit: bool
    pep: bool
    aml_risk: Literal["low", "medium", "high"]
    checks: list[ComplianceCheck]
    requires_human_review: bool = False


class AuditEntry(Strict):
    ts: datetime = Field(default_factory=utcnow)
    actor: str  # node / human id
    action: str
    detail: str
    message_id: str | None = None
    checksum: str | None = None


class HumanReviewRecord(Strict):
    reviewer: str
    decision: Literal["APPROVED", "REJECTED"]
    notes: str
    reviewed_at: datetime = Field(default_factory=utcnow)


class FinalDecision(Strict):
    decision_id: str = Field(default_factory=lambda: new_id("dec"))
    run_id: str
    applicant_id: str
    document_id: str
    decision: Decision
    reasons: list[str] = Field(min_length=1)
    conditions: list[str] = Field(default_factory=list)
    approved_amount_inr: float | None = Field(default=None, ge=0)
    interest_rate_pct: float | None = Field(default=None, ge=0, le=36)
    tenure_months: int | None = None
    risk_grade: RiskGrade | None = None
    credit_score: int | None = None
    summary: str = Field(description="Plain-language explanation for the customer/officer")
    human_review: HumanReviewRecord | None = None
    policy_version: str = POLICY_VERSION
    generated_at: datetime = Field(default_factory=utcnow)
    audit_trail: list[AuditEntry] = Field(default_factory=list)
    report_sha256: str | None = None

    @model_validator(mode="after")
    def _approval_fields(self) -> FinalDecision:
        if self.decision == Decision.APPROVED and (self.approved_amount_inr is None or self.interest_rate_pct is None):
            raise ValueError("APPROVED decisions must carry approved_amount_inr and interest_rate_pct")
        if self.decision == Decision.REJECTED and self.approved_amount_inr not in (None, 0):
            raise ValueError("REJECTED decisions must not carry an approved amount")
        return self


NODE_OUTPUT_SCHEMAS: dict[str, type[Strict]] = {
    "fetch_document": FetchDocumentOutput,
    "validate_document": ValidateDocumentOutput,
    "check_credit": CheckCreditOutput,
    "assess_risk": AssessRiskOutput,
    "check_compliance": CheckComplianceOutput,
    "compile_report": FinalDecision,
    "compile_rejection": FinalDecision,
}


# ---------------------------------------------------------------- traces
class ToolCallRecord(Strict):
    tool: str
    server: str
    arguments: dict[str, Any]
    ok: bool
    latency_ms: float
    error: str | None = None
    output_preview: str | None = None


class ReActStep(Strict):
    step: int
    thought: str
    action: str | None = None  # tool name or "finish"
    action_input: dict[str, Any] | None = None
    observation: str | None = None


class ErrorInfo(Strict):
    code: str
    message: str
    retryable: bool = False


class RunMetrics(Strict):
    latency_ms: float = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0
    tool_calls: int = 0


# ---------------------------------------------------------------- envelopes
class Envelope(Strict):
    schema_version: str = SCHEMA_VERSION
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    run_id: str
    created_at: datetime = Field(default_factory=utcnow)
    checksum: str | None = None  # sha256 of canonical body
    signature: str | None = None  # HMAC-SHA256 of checksum


class AgentTask(Envelope):
    sender: Literal["supervisor"] = "supervisor"
    recipient: str
    applicant_id: str = Field(pattern=r"^A\d{4,10}$")
    document_id: str = Field(pattern=r"^D-[A-Z0-9-]{3,40}$")
    instruction: str = Field(max_length=2000, description="Supervisor's directive for this subagent")
    context: dict[str, Any] = Field(default_factory=dict, description="Validated outputs of upstream nodes")
    deadline_s: float = Field(default=60.0, gt=0, le=600)
    attempt: int = Field(default=1, ge=1, le=10)

    @field_validator("recipient")
    @classmethod
    def _known_agent(cls, v: str) -> str:
        if v not in AGENT_NODES:
            raise ValueError(f"unknown recipient agent '{v}'")
        return v


class AgentResult(Envelope):
    in_reply_to: str
    sender: str
    status: Literal["success", "failed", "needs_review"]
    output: dict[str, Any] | None = None
    reasoning_trace: list[ReActStep] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    error: ErrorInfo | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)

    @model_validator(mode="after")
    def _status_consistency(self) -> AgentResult:
        if self.status == "success" and self.output is None:
            raise ValueError("success result must include output")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed result must include error")
        return self


# ---------------------------------------------------------------- supervisor
class RoutingDecision(Strict):
    """Chain-of-Thought output of the supervisor ("brain")."""
    thoughts: list[str] = Field(min_length=1, max_length=12, description="Step-by-step reasoning")
    next_node: NodeName
    instruction: str = Field(default="", max_length=1000, description="Directive passed to the next subagent")
    confidence: float = Field(ge=0, le=1)
    source: Literal["llm", "rules", "cache"] = "llm"


class RunStatus(Strict):
    run_id: str
    status: Literal["queued", "running", "awaiting_human", "completed", "failed"]
    applicant_id: str
    document_id: str
    current_node: str | None = None
    trajectory: list[str] = Field(default_factory=list)
    decision: FinalDecision | None = None
    pending_review: dict[str, Any] | None = None
    error: ErrorInfo | None = None
    updated_at: datetime = Field(default_factory=utcnow)
