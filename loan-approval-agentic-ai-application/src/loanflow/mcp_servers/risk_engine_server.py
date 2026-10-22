"""MCP server: Credit-risk rule engine & pricing (deterministic, versioned, auditable).

The *decision math lives here*, not in an LLM: agents call these tools and explain
the result. This keeps decisions reproducible for RBI audit / model-risk review.

Tools
  get_base_rate(loan_type)                                    -> product base rate (% p.a.)
  calculate_emi(principal_inr, annual_rate_pct, tenure_months)-> EMI (reducing balance)
  compute_risk_score(...)                                     -> grade A-E, PD proxy, DTI, LTV, hard stops

Run:  uvicorn loanflow.mcp_servers.risk_engine_server:app --port 9103
"""
from __future__ import annotations

import math

from loanflow.common.config import get_settings
from loanflow.common.schemas import POLICY_VERSION
from loanflow.mcp_layer.compat import create_server
from loanflow.mcp_servers._base import secure_asgi, tool_error

mcp = create_server("risk_engine", "Deterministic credit-risk scoring and loan pricing tools.")

BASE_RATES = {"home loan": 8.5, "personal loan": 12.5, "business loan": 13.5, "auto loan": 9.5}
GRADE_SPREAD = {"A": 0.0, "B": 1.0, "C": 2.5, "D": 4.0, "E": 6.0}
RBI_LTV_CAPS = [(3_000_000, 0.90), (7_500_000, 0.80), (math.inf, 0.75)]


def _emi(p: float, rate: float, n: int) -> float:
    r = rate / 1200
    if r == 0:
        return p / n
    return p * r * (1 + r) ** n / ((1 + r) ** n - 1)


def _grade(pd: float) -> str:
    for g, cut in (("A", 0.05), ("B", 0.12), ("C", 0.25), ("D", 0.40)):
        if pd < cut:
            return g
    return "E"


@mcp.tool()
async def get_base_rate(loan_type: str) -> dict:
    """Return the bank's current base lending rate (% p.a.) for a product."""
    rate = BASE_RATES.get(loan_type.strip().lower())
    if rate is None:
        return tool_error("UNKNOWN_PRODUCT", f"no base rate for '{loan_type}'")
    return {"ok": True, "loan_type": loan_type, "base_rate_pct": rate, "policy_version": POLICY_VERSION}


@mcp.tool()
async def calculate_emi(principal_inr: float, annual_rate_pct: float, tenure_months: int) -> dict:
    """Compute the monthly EMI on a reducing-balance basis."""
    if principal_inr <= 0 or tenure_months <= 0 or not 0 <= annual_rate_pct <= 36:
        return tool_error("INVALID_INPUT", "principal>0, tenure>0, 0<=rate<=36 required")
    emi = _emi(principal_inr, annual_rate_pct, tenure_months)
    return {"ok": True, "emi_inr": round(emi, 2), "total_payable_inr": round(emi * tenure_months, 2)}


@mcp.tool()
async def compute_risk_score(
    credit_score: int,
    monthly_income_inr: float,
    existing_emi_inr: float,
    loan_amount_inr: float,
    tenure_months: int,
    loan_type: str,
    employment_type: str,
    defaults_last_24m: int = 0,
    enquiries_last_6m: int = 0,
    credit_utilisation_pct: float = 0.0,
    property_value_inr: float | None = None,
) -> dict:
    """Score application risk. Returns grade (A best .. E worst), PD proxy, DTI, LTV, pricing and hard stops."""
    s = get_settings()
    if monthly_income_inr <= 0:
        return tool_error("INVALID_INPUT", "monthly income must be positive")
    base = BASE_RATES.get(loan_type.strip().lower(), 13.5)
    # Two-pass: score at base rate, then price at grade-adjusted rate.
    pd = 0.6 / (1 + math.exp((credit_score - 620) / 40))
    prelim_emi = _emi(loan_amount_inr, base, tenure_months)
    dti = (existing_emi_inr + prelim_emi) / monthly_income_inr
    ltv = loan_amount_inr / property_value_inr if property_value_inr else None
    factors = [f"bureau score {credit_score}"]
    pd += 0.25 * max(0.0, dti - 0.35)
    if dti > 0.35:
        factors.append(f"elevated DTI {dti:.2f}")
    pd += 0.05 * defaults_last_24m
    if defaults_last_24m:
        factors.append(f"{defaults_last_24m} recent default(s)")
    pd += 0.01 * max(0, enquiries_last_6m - 3)
    pd += 0.0005 * max(0.0, credit_utilisation_pct - 50)
    if credit_utilisation_pct > 50:
        factors.append(f"utilisation {credit_utilisation_pct:.0f}%")
    if employment_type == "self_employed":
        pd += 0.03
        factors.append("self-employed income volatility")
    if ltv is not None and ltv > 0.8:
        pd += 0.03
        factors.append(f"high LTV {ltv:.2f}")
    pd = min(pd, 0.99)
    grade = _grade(pd)
    rate = base + GRADE_SPREAD[grade]
    emi = _emi(loan_amount_inr, rate, tenure_months)
    dti = (existing_emi_inr + emi) / monthly_income_inr

    hard_stops = []
    if credit_score < s.min_credit_score:
        hard_stops.append(f"credit score {credit_score} below policy minimum {s.min_credit_score}")
    if dti > s.max_dti:
        hard_stops.append(f"DTI {dti:.2f} exceeds policy maximum {s.max_dti:.2f}")
    if ltv is not None:
        cap = next(c for limit, c in RBI_LTV_CAPS if loan_amount_inr <= limit)
        if ltv > cap:
            hard_stops.append(f"LTV {ltv:.2f} exceeds RBI cap {cap:.2f}")
    if hard_stops or grade in ("D", "E"):
        rec = "REJECTED"
    elif grade == "C":
        rec = "REFERRED"
    else:
        rec = "APPROVED"
    return {
        "ok": True,
        "risk_grade": grade,
        "risk_score": round(pd, 4),
        "proposed_emi_inr": round(emi, 2),
        "dti": round(dti, 4),
        "ltv": round(ltv, 4) if ltv is not None else None,
        "suggested_rate_pct": round(rate, 2),
        "factors": factors,
        "hard_stops": hard_stops,
        "recommended_decision": rec,
        "policy_version": POLICY_VERSION,
    }


app = secure_asgi(mcp)
