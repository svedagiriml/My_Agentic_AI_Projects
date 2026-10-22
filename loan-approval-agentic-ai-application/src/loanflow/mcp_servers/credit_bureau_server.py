"""MCP server: Credit bureau gateway (CIBIL / Experian / CRIF stand-in).

Tools
  check_bureau_consent(applicant_id)               -> whether a signed bureau-pull consent exists
  get_credit_report(applicant_id, pan_last4)       -> score, tradelines summary, flags

A real deployment calls the bureau's API through this server; credentials stay in
this service only (Secret Manager), never in agent prompts.

Run:  uvicorn loanflow.mcp_servers.credit_bureau_server:app --port 9102
"""
from __future__ import annotations

from loanflow.common.config import get_settings
from loanflow.mcp_layer.compat import create_server
from loanflow.mcp_servers._base import fixtures, secure_asgi, tool_error

mcp = create_server("credit_bureau", "Credit bureau tools. Pull requires consent; responses contain no raw PII.")


@mcp.tool()
async def check_bureau_consent(applicant_id: str) -> dict:
    """Check that the applicant has consented to a credit bureau enquiry (required by CICRA 2005)."""
    if applicant_id not in fixtures()["credit_bureau"]:
        return tool_error("NOT_FOUND", f"unknown applicant {applicant_id}")
    return {"ok": True, "applicant_id": applicant_id, "consent": True, "consent_ref": f"CNS-{applicant_id}"}


@mcp.tool()
async def get_credit_report(applicant_id: str, pan_last4: str) -> dict:
    """Fetch the CIBIL credit report summary. pan_last4 must match the bureau record."""
    rec = fixtures()["credit_bureau"].get(applicant_id)
    if rec is None:
        return tool_error("NOT_FOUND", f"no bureau record for {applicant_id}")
    if rec["pan_last4"] != pan_last4:
        return tool_error("IDENTITY_MISMATCH", "PAN does not match bureau record")
    flags = []
    if rec["defaults_last_24m"]:
        flags.append(f"{rec['defaults_last_24m']} default(s) in last 24 months")
    if rec["enquiries_last_6m"] > 3:
        flags.append("high enquiry velocity")
    if rec["credit_utilisation_pct"] > 75:
        flags.append("very high credit utilisation")
    return {
        "ok": True,
        "bureau": "CIBIL",
        "credit_score": rec["credit_score"],
        "active_loans": rec["active_loans"],
        "defaults_last_24m": rec["defaults_last_24m"],
        "enquiries_last_6m": rec["enquiries_last_6m"],
        "credit_utilisation_pct": rec["credit_utilisation_pct"],
        "flags": flags,
        "meets_minimum": rec["credit_score"] >= get_settings().min_credit_score and rec["defaults_last_24m"] == 0,
    }


app = secure_asgi(mcp)
