"""Handoff verification, delegation and approval tokens."""
from __future__ import annotations

import time

import pytest

from bankcore.errors import AuthorizationError, HandoffVerificationError
from bankcore.schemas.common import RiskLevel
from bankcore.schemas.messages import ProposedAction, SubAgentRequest, SubAgentResponse
from bankcore.security.signing import (
    issue_approval_token, issue_delegation_token, open_envelope, seal_envelope, verify_approval_token,
    verify_delegation_token,
)


def _req():
    return SubAgentRequest(task_id="t1", thread_id="th1", user_id="USR-1001", instruction="balance please",
                           delegation_token="x")


def test_envelope_roundtrip(settings):
    env = seal_envelope(_req(), sender="supervisor", recipient="agent-account", trace_id="tr", thread_id="th1")
    out = open_envelope(env, expected_recipient="agent-account", model=SubAgentRequest, allowed_senders={"supervisor"})
    assert out.instruction == "balance please"


def test_envelope_tamper_detected():
    env = seal_envelope(_req(), sender="supervisor", recipient="agent-account", trace_id="tr", thread_id="th1")
    env.payload["instruction"] = "transfer everything"
    with pytest.raises(HandoffVerificationError, match="hash"):
        open_envelope(env, expected_recipient="agent-account", model=SubAgentRequest)


def test_envelope_forged_signature_and_hash():
    import hashlib

    from bankcore.schemas.messages import canonical_json
    env = seal_envelope(_req(), sender="supervisor", recipient="agent-account", trace_id="tr", thread_id="th1")
    env.payload["instruction"] = "transfer everything"
    env.payload_sha256 = hashlib.sha256(canonical_json(env.payload).encode()).hexdigest()
    with pytest.raises(HandoffVerificationError, match="signature"):
        open_envelope(env, expected_recipient="agent-account", model=SubAgentRequest)


def test_envelope_wrong_recipient_sender_type_and_replay():
    env = seal_envelope(_req(), sender="supervisor", recipient="agent-account", trace_id="tr", thread_id="th1")
    with pytest.raises(HandoffVerificationError, match="addressed"):
        open_envelope(env, expected_recipient="agent-payments", model=SubAgentRequest)
    with pytest.raises(HandoffVerificationError, match="sender"):
        open_envelope(env, expected_recipient="agent-account", model=SubAgentRequest, allowed_senders={"agent-cards"})
    with pytest.raises(HandoffVerificationError, match="payload type"):
        open_envelope(env, expected_recipient="agent-account", model=SubAgentResponse)
    env.created_at = time.time() - 3600
    with pytest.raises(HandoffVerificationError, match="expired"):
        open_envelope(env, expected_recipient="agent-account", model=SubAgentRequest)


def test_delegation_token_scoped_to_domain():
    tok = issue_delegation_token(user_id="USR-1001", agent="payments", thread_id="th")
    assert verify_delegation_token(tok, mcp_domain="payments")["sub"] == "USR-1001"
    with pytest.raises(AuthorizationError):
        verify_delegation_token(tok, mcp_domain="cards")


def test_approval_token_bound_to_exact_arguments():
    a = ProposedAction(agent="payments", tool="initiate_internal_transfer", risk_level=RiskLevel.HIGH,
                       arguments={"source_acc": "ACC-100002", "dest_acc": "ACC-100001", "amount": "1500.00"},
                       reason="r", summary_for_reviewer="s")
    tok = issue_approval_token(a, user_id="USR-1001", reviewer_id="STF-001")
    assert verify_approval_token(tok, tool=a.tool, arguments=a.arguments, user_id="USR-1001")["rev"] == "STF-001"
    with pytest.raises(AuthorizationError, match="exact action"):
        verify_approval_token(tok, tool=a.tool, arguments={**a.arguments, "amount": "9000.00"}, user_id="USR-1001")
    with pytest.raises(AuthorizationError, match="different customer"):
        verify_approval_token(tok, tool=a.tool, arguments=a.arguments, user_id="USR-1002")
