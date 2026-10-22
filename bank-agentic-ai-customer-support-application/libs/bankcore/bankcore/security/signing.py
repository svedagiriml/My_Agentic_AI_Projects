"""Cryptographic integrity for agent handoffs, user delegation and HITL approvals.

* Handoff envelopes  — HMAC-SHA256 over canonical JSON; verified for recipient, freshness
  (anti-replay window) and payload hash, then re-validated with the expected pydantic model.
* Delegation tokens  — short-lived JWT "agent X acts for user U on MCP server Y". MCP servers
  derive the customer identity ONLY from this token, never from LLM-generated arguments,
  so a prompt-injected agent cannot read another customer's account.
* Approval tokens    — issued when a human approves a ProposedAction; bound to the exact
  tool name + argument hash and single-use (jti). High-risk MCP tools refuse to run without one.
"""
from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from typing import TypeVar

import jwt
from pydantic import BaseModel, ValidationError

from bankcore.config import Settings, get_settings
from bankcore.errors import AuthorizationError, HandoffVerificationError
from bankcore.schemas.messages import HandoffEnvelope, ProposedAction, canonical_json

M = TypeVar("M", bound=BaseModel)


# ------------------------------------------------------------- envelopes ----
def _sign(secret: str, *parts: str) -> str:
    return hmac.new(secret.encode(), "|".join(parts).encode(), hashlib.sha256).hexdigest()


def seal_envelope(payload: BaseModel, *, sender: str, recipient: str, trace_id: str, thread_id: str,
                  settings: Settings | None = None) -> HandoffEnvelope:
    s = settings or get_settings()
    body = payload.model_dump(mode="json")
    digest = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    message_id, created = uuid.uuid4().hex, time.time()
    sig = _sign(s.handoff_hmac_secret.get_secret_value(), message_id, sender, recipient, f"{created:.6f}",
                type(payload).__name__, digest)
    return HandoffEnvelope(message_id=message_id, trace_id=trace_id, thread_id=thread_id, sender=sender,
                           recipient=recipient, created_at=created, payload_type=type(payload).__name__,
                           payload=body, payload_sha256=digest, signature=sig)


def open_envelope(env: HandoffEnvelope, *, expected_recipient: str, model: type[M],
                  allowed_senders: set[str] | None = None, settings: Settings | None = None) -> M:
    """Verify an envelope and return the validated payload. Raises HandoffVerificationError."""
    s = settings or get_settings()
    if env.recipient != expected_recipient:
        raise HandoffVerificationError(f"envelope addressed to {env.recipient}, not {expected_recipient}")
    if allowed_senders is not None and env.sender not in allowed_senders:
        raise HandoffVerificationError(f"sender {env.sender} not allowed")
    if abs(time.time() - env.created_at) > s.handoff_max_skew_s:
        raise HandoffVerificationError("envelope expired or clock skew too large (possible replay)")
    if env.payload_type != model.__name__:
        raise HandoffVerificationError(f"unexpected payload type {env.payload_type}")
    digest = hashlib.sha256(canonical_json(env.payload).encode()).hexdigest()
    if not hmac.compare_digest(digest, env.payload_sha256):
        raise HandoffVerificationError("payload hash mismatch (tampered)")
    expected = _sign(s.handoff_hmac_secret.get_secret_value(), env.message_id, env.sender, env.recipient,
                     f"{env.created_at:.6f}", env.payload_type, env.payload_sha256)
    if not hmac.compare_digest(expected, env.signature):
        raise HandoffVerificationError("bad envelope signature")
    try:
        return model.model_validate(env.payload)
    except ValidationError as e:
        raise HandoffVerificationError(f"payload schema invalid: {e.error_count()} errors") from e


# ------------------------------------------------------- delegation token ----
def issue_delegation_token(*, user_id: str, agent: str, thread_id: str, ttl_s: int = 300,
                           settings: Settings | None = None) -> str:
    s = settings or get_settings()
    now = int(time.time())
    return jwt.encode({"sub": user_id, "act": agent, "aud": f"mcp:{agent}", "thr": thread_id,
                       "iat": now, "exp": now + ttl_s, "jti": uuid.uuid4().hex},
                      s.delegation_secret.get_secret_value(), algorithm="HS256")


def verify_delegation_token(token: str, *, mcp_domain: str, settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    try:
        return jwt.decode(token, s.delegation_secret.get_secret_value(), algorithms=["HS256"],
                          audience=f"mcp:{mcp_domain}", options={"require": ["exp", "sub", "aud", "act"]})
    except jwt.PyJWTError as e:
        raise AuthorizationError(f"invalid delegation token: {e}") from e


# --------------------------------------------------------- approval token ----
def issue_approval_token(action: ProposedAction, *, user_id: str, reviewer_id: str, ttl_s: int = 900,
                         settings: Settings | None = None) -> str:
    s = settings or get_settings()
    now = int(time.time())
    return jwt.encode({"sub": user_id, "rev": reviewer_id, "aid": action.action_id, "tool": action.tool,
                       "agent": action.agent.value, "ah": action.args_sha256, "iat": now,
                       "exp": now + ttl_s, "jti": uuid.uuid4().hex},
                      s.approval_secret.get_secret_value(), algorithm="HS256")


def verify_approval_token(token: str, *, tool: str, arguments: dict, user_id: str,
                          settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    try:
        claims = jwt.decode(token, s.approval_secret.get_secret_value(), algorithms=["HS256"],
                            options={"require": ["exp", "sub", "tool", "ah", "jti"]})
    except jwt.PyJWTError as e:
        raise AuthorizationError(f"invalid approval token: {e}") from e
    digest = hashlib.sha256(canonical_json({"tool": tool, "arguments": arguments}).encode()).hexdigest()
    if claims["tool"] != tool or not hmac.compare_digest(claims["ah"], digest):
        raise AuthorizationError("approval token does not match this exact action")
    if claims["sub"] != user_id:
        raise AuthorizationError("approval token issued for a different customer")
    return claims
