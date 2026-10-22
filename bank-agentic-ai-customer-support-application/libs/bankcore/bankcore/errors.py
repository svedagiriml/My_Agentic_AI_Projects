"""Exception hierarchy.

Every error carries a stable machine-readable ``code`` and a ``user_message`` that is
safe to show to a customer (never stack traces, SQL, account numbers, etc.).
``retryable`` drives retry / circuit-breaker behaviour.
"""
from __future__ import annotations


class BankAgentError(Exception):
    code: str = "internal_error"
    http_status: int = 500
    retryable: bool = False
    user_message: str = "Something went wrong on our side. Please try again shortly."

    def __init__(self, message: str = "", *, code: str | None = None, user_message: str | None = None,
                 details: dict | None = None) -> None:
        super().__init__(message or self.user_message)
        if code:
            self.code = code
        if user_message:
            self.user_message = user_message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.user_message}


# ---- client / validation ---------------------------------------------------
class ValidationFailed(BankAgentError):
    code, http_status = "validation_failed", 422
    user_message = "The request could not be validated."


class AuthenticationError(BankAgentError):
    code, http_status = "unauthenticated", 401
    user_message = "Authentication is required."


class AuthorizationError(BankAgentError):
    code, http_status = "forbidden", 403
    user_message = "You are not allowed to perform this action."


class RateLimitExceeded(BankAgentError):
    code, http_status = "rate_limited", 429
    user_message = "Too many requests. Please wait a moment and try again."


class GuardrailViolation(BankAgentError):
    code, http_status = "guardrail_blocked", 400
    user_message = ("I can't help with that request. If you need assistance with your accounts, "
                    "please rephrase your question.")


class HandoffVerificationError(BankAgentError):
    code, http_status = "handoff_rejected", 400
    user_message = "Internal message verification failed."


class ToolNotPermitted(BankAgentError):
    """Raised when an agent tries to call a tool outside its MCP allow-list (state isolation)."""
    code, http_status = "tool_not_permitted", 403


class ApprovalRequired(BankAgentError):
    code, http_status = "approval_required", 202
    user_message = "This action needs approval from a bank officer."


# ---- domain ---------------------------------------------------------------
class BankingDomainError(BankAgentError):
    """Business-rule failures (insufficient funds, card already frozen...). Not retryable."""
    code, http_status = "banking_rule_violation", 400
    user_message = "The banking operation could not be completed."


class NotFound(BankingDomainError):
    code, http_status = "not_found", 404
    user_message = "The requested record was not found."


# ---- infrastructure (retryable) --------------------------------------------
class DependencyError(BankAgentError):
    code, http_status, retryable = "dependency_error", 503, True
    user_message = "A dependent service is temporarily unavailable."


class DependencyTimeout(DependencyError):
    code, http_status = "dependency_timeout", 504
    user_message = "The request took too long. Please try again."


class CircuitOpenError(DependencyError):
    code = "circuit_open"
    user_message = "This service is temporarily unavailable. Please try again in a minute."


class LLMError(DependencyError):
    code = "llm_error"


class LLMOutputParseError(BankAgentError):
    code, retryable = "llm_output_invalid", True
