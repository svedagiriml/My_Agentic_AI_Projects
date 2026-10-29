"""Generates docs/architecture.svg (and .png if cairosvg is installed).  python docs/make_diagram.py"""
from __future__ import annotations

from html import escape as _e
from pathlib import Path

W, H = 1500, 1060
C = {"ink": "#1f2937", "muted": "#6b7280", "line": "#94a3b8", "sup": "#eef2ff", "supb": "#4f46e5",
     "agent": "#ecfdf5", "agentb": "#059669", "mcp": "#fff7ed", "mcpb": "#ea580c", "mem": "#f0f9ff",
     "memb": "#0284c7", "risk": "#fef2f2", "riskb": "#dc2626", "obs": "#faf5ff", "obsb": "#7c3aed",
     "grey": "#f8fafc", "greyb": "#64748b"}
out: list[str] = []


def box(x, y, w, h, fill, stroke, title, lines=(), r=10, title_size=14, dash=False, bold=True):
    d = ' stroke-dasharray="6 4"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="1.6"{d}/>')
    out.append(f'<text x="{x + w / 2}" y="{y + 22}" text-anchor="middle" font-size="{title_size}" '
               f'font-weight="{700 if bold else 500}" fill="{C["ink"]}">{_e(title)}</text>')
    for i, ln in enumerate(lines):
        out.append(f'<text x="{x + w / 2}" y="{y + 42 + i * 16}" text-anchor="middle" font-size="11.5" '
                   f'fill="{C["muted"]}">{_e(ln)}</text>')


def arrow(x1, y1, x2, y2, color=None, label=None, dash=False, lx=None, ly=None):
    color = color or C["line"]
    d = ' stroke-dasharray="5 4"' if dash else ""
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.8" '
               f'marker-end="url(#ah)"{d}/>')
    if label:
        out.append(f'<text x="{lx if lx is not None else (x1 + x2) / 2 + 6}" y="{ly if ly is not None else (y1 + y2) / 2}" '
                   f'font-size="11" fill="{C["muted"]}">{_e(label)}</text>')


def text(x, y, s, size=12, color=None, anchor="start", weight=400):
    w = len(s) * size * (0.56 if weight >= 600 else 0.52)
    rx = x - w / 2 if anchor == "middle" else x
    bg = "#eef2ff" if 175 < y < 475 else "#ffffff"
    out.append(f'<rect x="{rx - 3}" y="{y - size}" width="{w + 6}" height="{size + 5}" fill="{bg}" opacity="0.95"/>')
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color or C["muted"]}" text-anchor="{anchor}" '
               f'font-weight="{weight}">{_e(s)}</text>')


# ---------------------------------------------------------------- header --
text(W / 2, 34, "Horizon Trust Bank — Multi-Agent Customer Support (LangGraph · MCP · FastAPI · Cloud Run)", 20,
     C["ink"], "middle", 700)
text(W / 2, 56, "Supervisor (Chain-of-Thought) → ReAct sub-agents → their own MCP servers only · HITL for high-risk "
     "actions · durable checkpoints", 12.5, C["muted"], "middle")

# ------------------------------------------------------------- top row ----
box(40, 80, 190, 70, C["grey"], C["greyb"], "Customer", ["web · mobile · chat"])
box(270, 80, 230, 70, C["grey"], C["greyb"], "HTTPS LB + Cloud Armor", ["WAF · TLS · DDoS · geo rules"])
arrow(230, 115, 268, 115)
box(1250, 80, 210, 70, C["risk"], C["riskb"], "Bank officer (HITL)", ["staff console · MFA · four-eyes"])

# ------------------------------------------------------- supervisor box ----
SX, SY, SW, SH = 40, 175, 1420, 300
out.append(f'<rect x="{SX}" y="{SY}" width="{SW}" height="{SH}" rx="14" fill="{C["sup"]}" stroke="{C["supb"]}" '
           f'stroke-width="2"/>')
text(SX + 16, SY + 26, "SUPERVISOR service (FastAPI + LangGraph StateGraph) — the brain", 15, C["supb"], weight=700)
text(SX + 16, SY + 45, "JWT auth · rate limits · deadline · PII redaction at ingress · AsyncPostgresSaver "
     "checkpoint after every node", 11.5)
arrow(385, 150, 385, SY + 60, label="POST /v1/chat", lx=392, ly=166)

nodes = [("1 guard_input", ["injection scan", "topic scope"]), ("2 load_memory", ["Redis STM", "pgvector LTM"]),
         ("3 plan (CoT)", ["routing cache →", "LLM / keyword"]), ("4 dispatch (loop)", ["one task / step", "signed handoff"]),
         ("5 human_approval", ["interrupt()", "resume on decision"]), ("6 synthesize", ["CoT composition", "of results"]),
         ("7 compliance", ["silent reviewer", "revise once"]), ("8 finalize", ["output guard", "memory + audit"])]
nx, ny, nw, nh, gap = SX + 20, SY + 70, 157, 78, 16
for i, (t, ls) in enumerate(nodes):
    x = nx + i * (nw + gap)
    risk = i == 4
    box(x, ny, nw, nh, "#ffffff" if not risk else C["risk"], C["supb"] if not risk else C["riskb"], t, ls,
        title_size=12.5)
    if i < len(nodes) - 1:
        arrow(x + nw, ny + nh / 2, x + nw + gap - 2, ny + nh / 2, C["supb"])
# HITL link
hx = nx + 4 * (nw + gap) + nw / 2
out.append(f'<path d="M {hx + 40} {ny} C {hx + 40} 150, 1300 175, 1330 152" fill="none" stroke="{C["riskb"]}" '
           f'stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#ahr)"/>')
text(hx + 70, 168, "approve / reject (approval token bound to args)", 11, C["riskb"])
# loop arrow dispatch
dx = nx + 3 * (nw + gap)
out.append(f'<path d="M {dx + 30} {ny + nh} C {dx + 30} {ny + nh + 30}, {dx + 125} {ny + nh + 30}, {dx + 125} '
           f'{ny + nh + 2}" fill="none" stroke="{C["supb"]}" stroke-width="1.5" marker-end="url(#ahb)"/>')
text(dx + 18, ny + nh + 42, "context switching: Account → Cards …", 11, C["supb"])

# inner info chips
chips = [("Context engineering: 7 sections", ["instructions · tools · output format", "LTM · facts · notes · user input"]),
         ("Semantic routing cache (Redis)", ["entity-masked embeddings", "single-intent · versioned"]),
         ("Resilience", ["circuit breakers · bulkheads", "retries+jitter · timeouts · fallbacks"]),
         ("Guardrails", ["prompt-injection · PII · canary", "grounding check · memory guard"])]
for i, (t, ls) in enumerate(chips):
    x = SX + 20 + i * 350
    box(x, SY + 200, 335, 82, "#ffffff", "#c7d2fe", t, ls, title_size=12.5, r=8)

# ----------------------------------------------------------- agents row ----
agents = [("Account & Balance", "read-heavy"), ("Payments & Transfers", "HITL ≥ $1,000"),
          ("Card Operations", "HITL ATM ≥ $2,000"), ("Fraud & Security", "HITL lock/flag"),
          ("Loans & Mortgages", "disclosures"), ("Investments & Wealth", "HITL ≥ $5,000"),
          ("FAQ / Knowledge (RAG)", "hybrid retrieval"), ("Compliance (silent)", "redact + audit")]
mcp_tools = [["get_account_balance", "fetch_transaction_history", "download_monthly_statement"],
             ["initiate_internal_transfer", "schedule_bill_pay", "verify_routing_number", "get_exchange_rates"],
             ["freeze_or_unfreeze_card", "report_lost_stolen_card", "order_replacement_card", "update_atm_withdrawal_limit"],
             ["flag_fraudulent_transaction", "lock_user_profile", "trigger_mfa_challenge", "fetch_active_sessions"],
             ["get_loan_product_rates", "calculate_amortization_schedule", "fetch_loan_application_status"],
             ["fetch_cd_maturation_rates", "get_portfolio_valuation", "execute_fund_subscription"],
             ["query_knowledge_base", "locate_nearest_atm_or_branch"],
             ["redact_pii", "audit_compliance_check"]]
AY, aw, ah, agap = 545, 166, 64, 10
AX = 40 + (1420 - (8 * aw + 7 * agap)) / 2
for i, (t, s) in enumerate(agents):
    x = AX + i * (aw + agap)
    high = "HITL" in s
    box(x, AY, aw, ah, C["agent"] if not high else "#fff1f2", C["agentb"] if not high else C["riskb"], t, [s],
        title_size=12)
    arrow(x + aw / 2, SY + SH, x + aw / 2, AY - 2, C["agentb"])

# ------------------------------------------------------------- mcp row -----
MY, mh = 685, 132
for i, tools in enumerate(mcp_tools):
    x = AX + i * (aw + agap)
    box(x, MY, aw, mh, C["mcp"], C["mcpb"], f"mcp-{['account', 'payments', 'cards', 'fraud', 'loans', 'investments', 'knowledge', 'compliance'][i]}",
        [], title_size=12)
    for j, tl in enumerate(tools):
        short = tl if len(tl) <= 24 else tl[:23] + "…"
        out.append(f'<text x="{x + 8}" y="{MY + 44 + j * 19}" font-size="10.5" font-family="ui-monospace,Menlo,'
                   f'monospace" fill="{C["ink"]}">• {_e(short)}</text>')
    arrow(x + aw / 2, AY + ah, x + aw / 2, MY - 2, C["mcpb"])

# -------------------------------------------------------- bottom row -------
BY = 850
box(AX, BY, 5 * (aw + agap) - agap, 64, C["grey"], C["greyb"], "Core banking systems of record",
    ["accounts · cards · payments rails · fraud engine · LOS · wealth platform (via gateway; mock in this repo)"])
box(AX + 6 * (aw + agap), BY, aw, 64, C["mem"], C["memb"], "Knowledge base", ["pgvector + BM25 (RRF)"])
for i in range(6):
    x = AX + i * (aw + agap) + aw / 2
    if i < 5:
        arrow(x, MY + mh, x, BY - 2, C["greyb"])
arrow(AX + 6 * (aw + agap) + aw / 2, MY + mh, AX + 6 * (aw + agap) + aw / 2, BY - 2, C["memb"])

# memory + infra row
IY = 940
infra = [("Redis (Memorystore)", ["short-term: history · notes · state", "routing cache · rate limits · HITL index"],
          C["mem"], C["memb"]),
         ("Postgres 16 + pgvector (Cloud SQL)", ["LTM: episodic · semantic · procedural", "LangGraph checkpoints · audit log"],
          C["mem"], C["memb"]),
         ("Free LLMs", ["Ollama / vLLM (prefix KV cache)", "Groq free tier fallback · keyword router"], C["grey"], C["greyb"]),
         ("Pub/Sub → memory-worker", ["memory consolidation (guarded)", "audit · HITL notifications · DLQ"], C["obs"],
          C["obsb"]),
         ("Observability", ["Langfuse traces (one trace id end-to-end)", "Prometheus/Grafana: cost · latency · SLOs"],
          C["obs"], C["obsb"])]
iw = (1420 - 4 * 14) / 5
for i, (t, ls, f, s) in enumerate(infra):
    box(40 + i * (iw + 14), IY, iw, 84, f, s, t, ls, title_size=12.5)

text(40, AY - 20, "SUB-AGENTS — 8 Cloud Run services, one image (AGENT_NAME) · ReAct loop: Thought → Action → "
     "Observation · pydantic-verified handoffs", 13, C["agentb"], weight=700)
text(SX + 16, SY + SH + 22, "HMAC-signed HandoffEnvelope + Cloud Run ID token + delegation token (sub=user, aud=mcp:domain)",
     11.5, C["agentb"])
text(40, MY - 18, "MCP SERVERS — 8 Cloud Run services (ingress internal) · IAM: only agent-X may invoke mcp-X · "
     "tool allow-list · approval tokens enforced server-side", 13, C["mcpb"], weight=700)
defs = (f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{C["line"]}"/></marker>'
        f'<marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{C["riskb"]}"/></marker>'
        f'<marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{C["supb"]}"/></marker></defs>')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
       f'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif"><rect width="100%" height="100%" fill="#ffffff"/>'
       + defs + "".join(out) + "</svg>")
here = Path(__file__).parent
(here / "architecture.svg").write_text(svg, encoding="utf-8")
try:
    import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(here / "architecture.png"), output_width=W * 1.5)
except Exception:  # noqa: BLE001
    pass
print("wrote", here / "architecture.svg")
