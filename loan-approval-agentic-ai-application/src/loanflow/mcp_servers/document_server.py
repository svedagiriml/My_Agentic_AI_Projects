"""MCP server: Document Management System (DMS) + KYC.

Tools
  get_document(applicant_id, document_id)      -> PII-redacted text, sha256, injection flags
  extract_application_fields(applicant_id, document_id) -> structured fields (IDP/OCR stand-in)
  verify_kyc(applicant_id)                     -> PAN / CKYC / Aadhaar e-sign status

Run:  uvicorn loanflow.mcp_servers.document_server:app --port 9101
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from loanflow.mcp_layer.compat import create_server
from loanflow.mcp_servers._base import fixtures, secure_asgi, tool_error
from loanflow.security.guardrails import detect_injection, neutralize_injection, redact_pii

mcp = create_server("document", "Bank document management and KYC tools. Read-only.")

_FIELD_MAP = {
    "applicant_name": r"Applicant Name:\s*(.+)",
    "employment_type": r"Employment Type:\s*(.+)",
    "monthly_income_inr": r"Net Monthly Income \(INR\):\s*([\d,]+)",
    "existing_emi_inr": r"Existing EMIs \(INR/month\):\s*([\d,]+)",
    "loan_type": r"Loan Type:\s*(.+)",
    "loan_amount_inr": r"Loan Amount Requested \(INR\):\s*([\d,]+)",
    "tenure_months": r"Tenure \(months\):\s*(\d+)",
    "property_value_inr": r"Property Value \(INR\):\s*([\d,]+)",
    "loan_purpose": r"Purpose:\s*(.+)",
    "pan": r"PAN:\s*([A-Z]{5}\d{4}[A-Z])",
    "declaration_signed": r"Signature:\s*\[(signed)\]",
}
_NUMERIC = {"monthly_income_inr", "existing_emi_inr", "loan_amount_inr", "property_value_inr"}


def _load(applicant_id: str, document_id: str) -> dict[str, Any] | None:
    doc = fixtures()["documents"].get(document_id)
    if doc is None:
        return None
    return doc


@mcp.tool()
async def get_document(applicant_id: str, document_id: str) -> dict:
    """Fetch a loan document's text for an applicant. Text is PII-redacted and scanned for prompt injection."""
    doc = _load(applicant_id, document_id)
    if doc is None:
        return tool_error("NOT_FOUND", f"document {document_id} not found")
    if doc["applicant_id"] != applicant_id:  # object-level authorisation (prevents IDOR)
        return tool_error("FORBIDDEN", "document does not belong to this applicant")
    raw = doc["content"]
    inj = detect_injection(raw)
    text = neutralize_injection(raw) if inj.is_suspicious else raw
    red = redact_pii(text)
    return {
        "ok": True,
        "document_id": document_id,
        "applicant_id": applicant_id,
        "doc_type": doc["doc_type"],
        "content": red.text,
        "content_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "pii_redacted": red.found,
        "injection_flags": inj.matches,
    }


@mcp.tool()
async def extract_application_fields(applicant_id: str, document_id: str) -> dict:
    """Extract structured loan-application fields from the document (server-side, on the original)."""
    doc = _load(applicant_id, document_id)
    if doc is None:
        return tool_error("NOT_FOUND", f"document {document_id} not found")
    if doc["applicant_id"] != applicant_id:
        return tool_error("FORBIDDEN", "document does not belong to this applicant")
    fields: dict[str, Any] = {}
    for key, pat in _FIELD_MAP.items():
        m = re.search(pat, doc["content"])
        if not m:
            continue
        val = m.group(1).strip()
        if key in _NUMERIC:
            fields[key] = float(val.replace(",", ""))
        elif key == "tenure_months":
            fields[key] = int(val)
        elif key == "pan":
            fields["pan_last4"] = val[5:9]  # never return the full PAN
        elif key == "employment_type":
            v = val.lower().replace("-", "_").replace(" ", "_")
            fields[key] = v if v in {"salaried", "self_employed", "unemployed", "retired"} else "unknown"
        else:
            fields[key] = val
    required = ["applicant_name", "employment_type", "monthly_income_inr", "existing_emi_inr",
                "loan_amount_inr", "tenure_months", "loan_type", "loan_purpose", "declaration_signed"]
    missing = [f for f in required if f not in fields]
    return {"ok": True, "fields": fields, "missing_fields": missing}


@mcp.tool()
async def verify_kyc(applicant_id: str) -> dict:
    """Verify KYC: PAN validity (NSDL), CKYC registry record and Aadhaar e-sign."""
    rec = fixtures()["kyc"].get(applicant_id)
    if rec is None:
        return tool_error("NOT_FOUND", f"no KYC record for {applicant_id}")
    verified = all(rec.values())
    return {"ok": True, "applicant_id": applicant_id, **rec, "kyc_verified": verified}


app = secure_asgi(mcp)
