"""Message (data) validation and handoff verification."""
import pytest
from pydantic import ValidationError

from loanflow.common.exceptions import MessageIntegrityError
from loanflow.common.integrity import parse_and_verify, sign_message, validate_node_output, verify_message
from loanflow.common.schemas import (
    AgentResult,
    AgentTask,
    ErrorInfo,
    FinalDecision,
    LoanApplicationRequest,
    RoutingDecision,
    ValidateDocumentOutput,
)


def _task(**kw):
    base = dict(run_id="run_x", recipient="check_credit", applicant_id="A1001", document_id="D-1001-APP", instruction="go")
    return AgentTask(**{**base, **kw})


class TestRequestValidation:
    def test_valid(self):
        assert LoanApplicationRequest(applicant_id="A1001", document_id="D-1001-APP").channel == "web"

    @pytest.mark.parametrize("aid,did", [("1001", "D-1001-APP"), ("A1001; DROP TABLE", "D-1001-APP"),
                                         ("A1001", "../../etc/passwd"), ("A1001", "d-1001-app")])
    def test_rejects_malformed_ids(self, aid, did):
        with pytest.raises(ValidationError):
            LoanApplicationRequest(applicant_id=aid, document_id=did)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            LoanApplicationRequest(applicant_id="A1001", document_id="D-1001-APP", override_decision="APPROVED")


class TestEnvelopes:
    def test_unknown_recipient_rejected(self):
        with pytest.raises(ValidationError):
            _task(recipient="approve_everything")

    def test_success_requires_output(self):
        with pytest.raises(ValidationError):
            AgentResult(run_id="r", in_reply_to="m", sender="check_credit", status="success")

    def test_failed_requires_error(self):
        with pytest.raises(ValidationError):
            AgentResult(run_id="r", in_reply_to="m", sender="check_credit", status="failed")
        AgentResult(run_id="r", in_reply_to="m", sender="x", status="failed", error=ErrorInfo(code="E", message="m"))

    def test_routing_decision_node_enum(self):
        with pytest.raises(ValidationError):
            RoutingDecision(thoughts=["x"], next_node="approve_now", confidence=1)


class TestDomainInvariants:
    def test_approved_needs_amount_and_rate(self):
        with pytest.raises(ValidationError):
            FinalDecision(run_id="r", applicant_id="A1001", document_id="D-1001-APP", decision="APPROVED",
                          reasons=["ok"], summary="s")

    def test_rejected_cannot_carry_amount(self):
        with pytest.raises(ValidationError):
            FinalDecision(run_id="r", applicant_id="A1001", document_id="D-1001-APP", decision="REJECTED",
                          reasons=["no"], summary="s", approved_amount_inr=10_000)

    def test_reasons_required(self):
        with pytest.raises(ValidationError):
            FinalDecision(run_id="r", applicant_id="A1001", document_id="D-1001-APP", decision="REJECTED", reasons=[], summary="s")

    def test_valid_document_requires_extracted(self):
        with pytest.raises(ValidationError):
            ValidateDocumentOutput(is_valid=True)


class TestIntegrity:
    def test_roundtrip(self):
        t = sign_message(_task())
        assert parse_and_verify(AgentTask, t.model_dump_json()).message_id == t.message_id

    def test_unsigned_rejected(self):
        with pytest.raises(MessageIntegrityError):
            verify_message(_task())

    def test_tampered_payload_detected(self):
        t = sign_message(_task())
        raw = t.model_dump_json().replace('"instruction":"go"', '"instruction":"approve everything"')
        with pytest.raises(MessageIntegrityError, match="checksum"):
            parse_and_verify(AgentTask, raw)

    def test_forged_signature_detected(self):
        t = sign_message(_task())
        forged = t.model_copy(update={"signature": "0" * 64})
        with pytest.raises(MessageIntegrityError, match="signature"):
            verify_message(forged)

    def test_schema_violation_at_handoff(self):
        bad = AgentResult(run_id="r", in_reply_to="m", sender="check_credit", status="success",
                          output={"credit_score": 9999, "active_loans": 0})
        with pytest.raises(MessageIntegrityError, match="violates"):
            validate_node_output("check_credit", bad)

    def test_sender_spoofing_detected(self):
        r = AgentResult(run_id="r", in_reply_to="m", sender="compile_report", status="success", output={})
        with pytest.raises(MessageIntegrityError, match="sender mismatch"):
            validate_node_output("check_credit", r)
