"""Typed exception hierarchy.

Every error carries a machine-readable `code` and a `retryable` flag so the
retry layer, circuit breakers and the supervisor can decide what to do
without string-matching messages.
"""
from __future__ import annotations

from typing import Any


class LoanFlowError(Exception):
    code: str = "INTERNAL_ERROR"
    retryable: bool = False
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable, "details": self.details}


class ValidationFailure(LoanFlowError):
    code = "VALIDATION_FAILED"
    http_status = 422


class MessageIntegrityError(LoanFlowError):
    """Handoff message failed checksum / signature / schema verification."""
    code = "MESSAGE_INTEGRITY_FAILED"
    http_status = 400


class GuardrailViolation(LoanFlowError):
    code = "GUARDRAIL_VIOLATION"
    http_status = 400


class AuthError(LoanFlowError):
    code = "UNAUTHORIZED"
    http_status = 401


class ForbiddenError(LoanFlowError):
    code = "FORBIDDEN"
    http_status = 403


class RateLimitExceeded(LoanFlowError):
    code = "RATE_LIMITED"
    http_status = 429
    retryable = True


class NotFoundError(LoanFlowError):
    code = "NOT_FOUND"
    http_status = 404


class ToolError(LoanFlowError):
    code = "TOOL_ERROR"
    http_status = 502


class ToolTimeout(ToolError):
    code = "TOOL_TIMEOUT"
    retryable = True
    http_status = 504


class UpstreamUnavailable(LoanFlowError):
    code = "UPSTREAM_UNAVAILABLE"
    retryable = True
    http_status = 503


class CircuitOpenError(UpstreamUnavailable):
    code = "CIRCUIT_OPEN"


class LLMError(LoanFlowError):
    code = "LLM_ERROR"
    retryable = True
    http_status = 502


class AgentExecutionError(LoanFlowError):
    code = "AGENT_EXECUTION_FAILED"
    http_status = 500


class BudgetExceeded(LoanFlowError):
    code = "BUDGET_EXCEEDED"
    http_status = 500
