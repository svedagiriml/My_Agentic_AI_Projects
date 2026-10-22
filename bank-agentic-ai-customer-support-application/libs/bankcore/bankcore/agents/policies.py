"""Risk & Human-in-the-Loop policy — ONE definition shared by sub-agents and MCP servers.

Sub-agents use it to *propose* (pause) instead of executing; MCP servers use it to *enforce*
(refuse to execute without a valid, argument-bound approval token). Defence in depth: even a
compromised sub-agent cannot move money above the threshold without a human.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from bankcore.config import Settings, get_settings
from bankcore.schemas.common import RiskLevel


@dataclass(frozen=True)
class ApprovalRequirement:
    required: bool
    risk: RiskLevel
    reason: str


def _amount(args: dict[str, Any], key: str) -> Decimal:
    try:
        return Decimal(str(args.get(key, "0")))
    except (InvalidOperation, TypeError):
        return Decimal("0")


# Tools that change state. They are never blindly retried (idempotency keys instead).
MUTATING_TOOLS = {
    "initiate_internal_transfer", "schedule_bill_pay", "freeze_or_unfreeze_card", "report_lost_stolen_card",
    "order_replacement_card", "update_atm_withdrawal_limit", "flag_fraudulent_transaction", "lock_user_profile",
    "trigger_mfa_challenge", "execute_fund_subscription",
}


def approval_requirement(tool: str, args: dict[str, Any], settings: Settings | None = None) -> ApprovalRequirement:
    s = settings or get_settings()
    if tool == "initiate_internal_transfer":
        amt = _amount(args, "amount")
        if amt >= Decimal(str(s.hitl_transfer_threshold)):
            return ApprovalRequirement(True, RiskLevel.HIGH, f"transfer of {amt} >= HITL threshold")
        return ApprovalRequirement(False, RiskLevel.MEDIUM, "transfer below threshold")
    if tool == "schedule_bill_pay":
        amt = _amount(args, "amount")
        if amt >= Decimal(str(s.hitl_transfer_threshold)):
            return ApprovalRequirement(True, RiskLevel.HIGH, f"bill payment of {amt} >= HITL threshold")
        return ApprovalRequirement(False, RiskLevel.MEDIUM, "bill payment below threshold")
    if tool == "lock_user_profile":
        return ApprovalRequirement(True, RiskLevel.CRITICAL, "destructive: locks profile and revokes all sessions")
    if tool == "flag_fraudulent_transaction":
        return ApprovalRequirement(True, RiskLevel.HIGH, "opens fraud case / provisional credit (financial impact)")
    if tool == "update_atm_withdrawal_limit":
        amt = _amount(args, "new_limit")
        if amt >= Decimal(str(s.hitl_atm_limit_threshold)):
            return ApprovalRequirement(True, RiskLevel.HIGH, f"ATM limit {amt} >= HITL threshold")
        return ApprovalRequirement(False, RiskLevel.MEDIUM, "limit change below threshold")
    if tool == "execute_fund_subscription":
        amt = _amount(args, "amount")
        if amt >= Decimal(str(s.hitl_fund_subscription_threshold)):
            return ApprovalRequirement(True, RiskLevel.HIGH, f"investment of {amt} >= HITL threshold")
        return ApprovalRequirement(False, RiskLevel.MEDIUM, "investment below threshold")
    if tool in MUTATING_TOOLS:
        return ApprovalRequirement(False, RiskLevel.MEDIUM, "state-changing, customer-initiated")
    return ApprovalRequirement(False, RiskLevel.LOW, "read-only")


def reviewer_summary(tool: str, args: dict[str, Any]) -> str:
    pretty = ", ".join(f"{k}={v}" for k, v in sorted(args.items()))
    return f"{tool}({pretty})"
