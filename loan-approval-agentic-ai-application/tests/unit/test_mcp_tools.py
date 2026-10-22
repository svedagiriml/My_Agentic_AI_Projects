"""Tool outputs, exercised through the real MCP protocol (in-process transport)."""
import pytest

from loanflow.common.exceptions import ForbiddenError
from loanflow.mcp_layer.client import MCPToolbox
from loanflow.mcp_layer.compat import connect, parse_tool_result
from loanflow.mcp_servers import compliance_server, credit_bureau_server, document_server, risk_engine_server


async def call(server, tool, args):
    async with connect(server) as c:
        return parse_tool_result(await c.call_tool(tool, args))


class TestDocumentServer:
    async def test_get_document_redacts_and_hashes(self):
        err, d = await call(document_server.mcp, "get_document", {"applicant_id": "A1001", "document_id": "D-1001-APP"})
        assert not err and d["ok"]
        assert "ABCPK1234F" not in d["content"] and "PAN" in d["pii_redacted"]
        assert len(d["content_sha256"]) == 64 and d["injection_flags"] == []

    async def test_idor_forbidden(self):
        _, d = await call(document_server.mcp, "get_document", {"applicant_id": "A1002", "document_id": "D-1001-APP"})
        assert d["ok"] is False and d["error"]["code"] == "FORBIDDEN"

    async def test_injection_flagged_and_neutralised(self):
        _, d = await call(document_server.mcp, "get_document", {"applicant_id": "A1006", "document_id": "D-1006-APP"})
        assert "override_instructions" in d["injection_flags"]
        assert "ignore all previous instructions" not in d["content"].lower()

    async def test_extract_fields_and_missing(self):
        _, d = await call(document_server.mcp, "extract_application_fields", {"applicant_id": "A1001", "document_id": "D-1001-APP"})
        assert d["fields"]["monthly_income_inr"] == 150000 and d["fields"]["pan_last4"] == "1234" and not d["missing_fields"]
        _, d = await call(document_server.mcp, "extract_application_fields", {"applicant_id": "A1007", "document_id": "D-1007-APP"})
        assert "monthly_income_inr" in d["missing_fields"]

    async def test_bad_arguments_are_protocol_errors(self):
        err, _ = await call(document_server.mcp, "get_document", {"applicant_id": "A1001"})
        assert err


class TestCreditBureau:
    async def test_report(self):
        _, d = await call(credit_bureau_server.mcp, "get_credit_report", {"applicant_id": "A1003", "pan_last4": "9012"})
        assert d["credit_score"] == 590 and d["meets_minimum"] is False and any("default" in f for f in d["flags"])

    async def test_identity_mismatch(self):
        _, d = await call(credit_bureau_server.mcp, "get_credit_report", {"applicant_id": "A1001", "pan_last4": "0000"})
        assert d["error"]["code"] == "IDENTITY_MISMATCH"


class TestRiskEngine:
    async def test_emi_matches_formula(self):
        _, d = await call(risk_engine_server.mcp, "calculate_emi", {"principal_inr": 100000, "annual_rate_pct": 12, "tenure_months": 12})
        assert d["emi_inr"] == pytest.approx(8884.88, abs=0.01)

    async def test_invalid_emi_input(self):
        _, d = await call(risk_engine_server.mcp, "calculate_emi", {"principal_inr": -5, "annual_rate_pct": 12, "tenure_months": 12})
        assert d["ok"] is False

    @pytest.mark.parametrize("score,income,emi,expected_grade,expected_rec", [
        (782, 150000, 15000, "A", "APPROVED"), (668, 120000, 20000, "C", "REFERRED"), (590, 90000, 10000, "E", "REJECTED"),
    ])
    async def test_grades(self, score, income, emi, expected_grade, expected_rec):
        _, d = await call(risk_engine_server.mcp, "compute_risk_score", {
            "credit_score": score, "monthly_income_inr": income, "existing_emi_inr": emi, "loan_amount_inr": 1200000,
            "tenure_months": 60, "loan_type": "Personal Loan",
            "employment_type": "self_employed" if score == 668 else "salaried", "credit_utilisation_pct": 55.0})
        assert d["risk_grade"] == expected_grade and d["recommended_decision"] == expected_rec

    async def test_dti_hard_stop(self):
        _, d = await call(risk_engine_server.mcp, "compute_risk_score", {
            "credit_score": 760, "monthly_income_inr": 60000, "existing_emi_inr": 25000, "loan_amount_inr": 800000,
            "tenure_months": 36, "loan_type": "Personal Loan", "employment_type": "salaried"})
        assert d["recommended_decision"] == "REJECTED" and any("DTI" in h for h in d["hard_stops"])


class TestCompliance:
    async def test_sanctions_fuzzy_hit(self):
        _, d = await call(compliance_server.mcp, "screen_sanctions", {"applicant_name": "Viktor Petrov Sanko"})
        assert d["sanctions_hit"] and d["matched_list"] == "OFAC SDN"
        _, d = await call(compliance_server.mcp, "screen_sanctions", {"applicant_name": "Victor Petrov Sanko"})
        assert d["sanctions_hit"]  # one-letter variant still caught

    async def test_clean_name(self):
        _, d = await call(compliance_server.mcp, "screen_sanctions", {"applicant_name": "Ravi Kumar"})
        assert not d["sanctions_hit"]

    async def test_rbi_ltv(self):
        _, d = await call(compliance_server.mcp, "check_lending_regulations",
                          {"loan_type": "Home Loan", "loan_amount_inr": 9_000_000, "tenure_months": 300, "property_value_inr": 10_000_000})
        assert d["all_passed"] is False  # 90% LTV > 75% cap above 75 lakh


async def test_toolbox_enforces_allow_list():
    async with MCPToolbox({"document": ["get_document"]}) as tb:
        assert [t["name"] for t in tb.catalog] == ["get_document"]
        with pytest.raises(ForbiddenError):
            await tb.call("verify_kyc", {"applicant_id": "A1001"})
        out = await tb.call("get_document", {"applicant_id": "A1001", "document_id": "D-1001-APP"})
        assert out["ok"] and tb.records[-1].ok and "ABCPK1234F" not in (tb.records[-1].output_preview or "")
