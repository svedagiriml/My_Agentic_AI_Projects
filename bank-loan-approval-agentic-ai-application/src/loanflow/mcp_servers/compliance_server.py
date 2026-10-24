"""MCP server: Compliance (AML/CFT, sanctions, PEP, RBI lending rules).

Tools
  screen_sanctions(applicant_name)                         -> UN / OFAC / MHA list screening (fuzzy)
  check_pep(applicant_name)                                -> politically exposed person screening
  assess_aml_risk(applicant_id, loan_amount_inr, monthly_income_inr, employment_type, loan_purpose)
  check_lending_regulations(loan_type, loan_amount_inr, property_value_inr, tenure_months)

Run:  uvicorn loanflow.mcp_servers.compliance_server:app --port 9104
"""
from __future__ import annotations

from difflib import SequenceMatcher

from loanflow.mcp_layer.compat import create_server
from loanflow.mcp_servers._base import fixtures, secure_asgi

mcp = create_server("compliance", "AML, sanctions, PEP and RBI regulatory checks.")

_HIGH_RISK_PURPOSE = ("import", "export", "crypto", "cash", "trading", "remittance")


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


@mcp.tool()
async def screen_sanctions(applicant_name: str) -> dict:
    """Screen a name against UN 1267, OFAC SDN and MHA (UAPA) sanctions lists using fuzzy matching."""
    best = max(fixtures()["watchlists"]["sanctions"], key=lambda e: _similar(applicant_name, e["name"]))
    score = _similar(applicant_name, best["name"])
    hit = score >= 0.85
    return {
        "ok": True,
        "sanctions_hit": hit,
        "match_score": round(score, 3),
        "matched_list": best["list"] if hit else None,
        "matched_ref": best["ref"] if hit else None,
        "regulation": "PMLA 2002 s.12; UAPA 1967 s.51A; RBI KYC Master Direction ch. X",
    }


@mcp.tool()
async def check_pep(applicant_name: str) -> dict:
    """Check whether the applicant is a Politically Exposed Person (enhanced due diligence required)."""
    peps = fixtures()["watchlists"]["pep"]
    match = next((p for p in peps if _similar(applicant_name, p["name"]) >= 0.9), None)
    return {"ok": True, "pep": match is not None, "position": match["position"] if match else None,
            "regulation": "RBI KYC Master Direction s.41 (PEP EDD)"}


@mcp.tool()
async def assess_aml_risk(
    applicant_id: str, loan_amount_inr: float, monthly_income_inr: float, employment_type: str, loan_purpose: str
) -> dict:
    """Rule-based AML risk categorisation (low/medium/high) per RBI risk-based approach."""
    reasons = []
    multiple = loan_amount_inr / max(monthly_income_inr, 1)
    if multiple > 120:
        reasons.append(f"loan is {multiple:.0f}x monthly income")
    if employment_type == "self_employed":
        reasons.append("self-employed: cash-intensive business risk")
    if any(w in loan_purpose.lower() for w in _HIGH_RISK_PURPOSE):
        reasons.append("high-risk purpose keyword")
    level = "high" if len(reasons) >= 2 else "medium" if reasons else "low"
    return {"ok": True, "applicant_id": applicant_id, "aml_risk": level, "reasons": reasons,
            "regulation": "PMLA Rules 2005; FIU-IND STR guidelines"}


@mcp.tool()
async def check_lending_regulations(
    loan_type: str, loan_amount_inr: float, tenure_months: int, property_value_inr: float | None = None
) -> dict:
    """Validate the loan against RBI lending norms (LTV caps for housing, tenure limits)."""
    checks = []
    lt = loan_type.lower()
    if "home" in lt:
        if property_value_inr is None:
            checks.append({"name": "ltv_cap", "passed": False, "detail": "property value missing", "regulation": "RBI Housing Finance LTV"})
        else:
            cap = 0.90 if loan_amount_inr <= 3_000_000 else 0.80 if loan_amount_inr <= 7_500_000 else 0.75
            ltv = loan_amount_inr / property_value_inr
            checks.append({"name": "ltv_cap", "passed": ltv <= cap, "detail": f"LTV {ltv:.2f} vs cap {cap:.2f}",
                           "regulation": "RBI Master Circular - Housing Finance"})
        checks.append({"name": "tenure_limit", "passed": tenure_months <= 360, "detail": f"{tenure_months}m <= 360m",
                       "regulation": "Bank board-approved policy"})
    else:
        checks.append({"name": "tenure_limit", "passed": tenure_months <= 84, "detail": f"{tenure_months}m <= 84m",
                       "regulation": "Bank board-approved policy (unsecured)"})
    checks.append({"name": "fair_practices_code", "passed": True, "detail": "KFS to be issued before disbursal",
                   "regulation": "RBI Fair Practices Code / Key Fact Statement circular 2024"})
    return {"ok": True, "checks": checks, "all_passed": all(c["passed"] for c in checks)}


app = secure_asgi(mcp)
