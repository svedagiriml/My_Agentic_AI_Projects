"""MCP server: Decision reporting, audit store and customer notification.

Tools
  render_decision_report(decision)     -> markdown report + sha256 (tamper evidence)
  store_decision(decision)             -> append-only audit record (WORM bucket / Postgres in prod)
  notify_applicant(applicant_id, decision_id, decision) -> queues SMS/e-mail (template only, no PII in payload)

Run:  uvicorn loanflow.mcp_servers.reporting_server:app --port 9105
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from loanflow.common.config import get_settings
from loanflow.mcp_layer.compat import create_server
from loanflow.mcp_servers._base import secure_asgi, tool_error

mcp = create_server("reporting", "Decision report rendering, immutable audit storage and notifications.")

AUDIT_LOG: list[dict] = []  # dev/test only; prod writes the decision_audit table (POSTGRES_DSN)
OUTBOX: list[dict] = []


@mcp.tool()
async def render_decision_report(decision: dict) -> dict:
    """Render the final decision as a markdown report and return its SHA-256 for tamper evidence."""
    required = {"decision", "applicant_id", "reasons", "run_id"}
    if not required <= decision.keys():
        return tool_error("INVALID_INPUT", f"decision missing {sorted(required - decision.keys())}")
    lines = [
        f"# Loan Decision Report — {decision['decision']}",
        f"- Run: `{decision['run_id']}`  Applicant: `{decision['applicant_id']}`  Document: `{decision.get('document_id')}`",
        f"- Policy version: {decision.get('policy_version')}  Generated: {datetime.now(UTC).isoformat()}",
    ]
    if decision.get("approved_amount_inr"):
        lines.append(
            f"- Approved: INR {decision['approved_amount_inr']:,.0f} @ {decision.get('interest_rate_pct')}% "
            f"for {decision.get('tenure_months')} months"
        )
    lines += ["", "## Reasons", *[f"- {r}" for r in decision["reasons"]]]
    if decision.get("conditions"):
        lines += ["", "## Conditions", *[f"- {c}" for c in decision["conditions"]]]
    if decision.get("summary"):
        lines += ["", "## Summary", decision["summary"]]
    md = "\n".join(lines)
    return {"ok": True, "report_markdown": md, "report_sha256": hashlib.sha256(md.encode()).hexdigest()}


_pool = None


async def _pg():
    global _pool
    dsn = get_settings().postgres_dsn
    if not dsn:
        return None
    if _pool is None:
        import asyncpg

        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5, command_timeout=5)
    return _pool


@mcp.tool()
async def store_decision(decision: dict) -> dict:
    """Persist the decision to the append-only audit store; returns the audit record id and hash chain link."""
    body = json.dumps(decision, sort_keys=True, default=str)
    pool = await _pg()
    if pool is not None:  # Postgres append-only, hash-chained table (db/init.sql)
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("LOCK TABLE decision_audit IN SHARE ROW EXCLUSIVE MODE")
            prev = await conn.fetchval("SELECT chain_hash FROM decision_audit ORDER BY stored_at DESC LIMIT 1") or "GENESIS"
            chain = hashlib.sha256((prev + body).encode()).hexdigest()
            audit_id = f"aud_{chain[:20]}"
            await conn.execute(
                "INSERT INTO decision_audit (audit_id, run_id, decision, prev_hash, chain_hash) VALUES ($1,$2,$3::jsonb,$4,$5)",
                audit_id, str(decision.get("run_id")), body, prev, chain)
        return {"ok": True, "audit_id": audit_id, "chain_hash": chain, "prev_hash": prev,
                "stored_at": datetime.now(UTC).isoformat()}
    prev = AUDIT_LOG[-1]["chain_hash"] if AUDIT_LOG else "GENESIS"  # dev/test fallback
    chain = hashlib.sha256((prev + body).encode()).hexdigest()
    rec = {"audit_id": f"aud_{len(AUDIT_LOG) + 1:08d}", "chain_hash": chain, "prev_hash": prev,
           "stored_at": datetime.now(UTC).isoformat()}
    AUDIT_LOG.append({**rec, "decision": decision})
    return {"ok": True, **rec}


@mcp.tool()
async def notify_applicant(applicant_id: str, decision_id: str, decision: str) -> dict:
    """Queue a templated decision notification to the applicant (contact details resolved downstream)."""
    if decision not in {"APPROVED", "REJECTED", "REFERRED"}:
        return tool_error("INVALID_INPUT", "unknown decision")
    msg = {"applicant_id": applicant_id, "decision_id": decision_id, "template": f"loan_decision_{decision.lower()}"}
    OUTBOX.append(msg)
    return {"ok": True, "queued": True, **msg}


app = secure_asgi(mcp)
