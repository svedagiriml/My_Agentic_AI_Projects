"""MCP server: Compliance, Auditing & Guardrails (silent reviewer)."""
from __future__ import annotations

from bankcore.guardrails.pii import redact_text
from bankcore.mcp.server import BankMCPServer, ToolContext
from bankcore.schemas import tools as T

from mcp_servers.compliance.rules import audit


def create(**kwargs) -> BankMCPServer:
    srv = BankMCPServer("compliance", "PII redaction and regulatory audit of draft responses.", **kwargs)

    @srv.tool(name="redact_pii", input_model=T.RedactInput, output_model=T.RedactOutput,
              description="Strip raw SSNs, full card numbers, account numbers, passwords, PINs, CVVs, e-mails, "
                          "phone numbers and national IDs from a text payload.")
    async def redact_pii(ctx: ToolContext, text_payload: str) -> dict:
        res = redact_text(text_payload)
        return {"redacted_text": res.text, "entities_found": res.entities}

    @srv.tool(name="audit_compliance_check", input_model=T.AuditInput, output_model=T.AuditOutput,
              description="Audit a draft customer response against Fair Lending (ECOA/Reg B), Truth in Savings "
                          "(Reg DD), TILA (Reg Z), Reg E, UDAAP, Reg BI and GDPR/CCPA/GLBA rules.")
    async def audit_compliance_check(ctx: ToolContext, agent_response_text: str) -> dict:
        return audit(agent_response_text)

    return srv
