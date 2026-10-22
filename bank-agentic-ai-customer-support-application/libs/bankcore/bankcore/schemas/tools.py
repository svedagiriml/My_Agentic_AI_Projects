"""Input/output contracts for every MCP tool.

MCP tool functions accept primitive arguments (clean JSON schema for the LLM) and
immediately validate them into the *Input models below. Outputs are validated on
the server (before returning) and again on the client (``MCPToolClient``) so a
compromised or buggy server cannot smuggle malformed data into an agent.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, model_validator

from bankcore.schemas.common import (
    AccountId, CardId, CurrencyCode, FundId, InvestmentAccountId, LoanApplicationId, Money,
    PayeeId, RoutingNumber, StrictModel, TransactionId, UserId,
)


class ToolError(StrictModel):
    code: str
    message: str


class ToolEnvelope(StrictModel):
    """Uniform result shape for every tool: never raise across the MCP boundary."""
    ok: bool
    tool: str
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    requires_approval: bool = False


# ---------------------------------------------------------------- accounts --
class GetBalanceInput(StrictModel):
    account_id: AccountId


class BalanceOutput(StrictModel):
    account_id: str
    account_type: Literal["checking", "savings", "credit"]
    currency: str
    available_balance: Decimal
    ledger_balance: Decimal
    credit_limit: Decimal | None = None
    as_of: datetime


class DateRange(StrictModel):
    start: date
    end: date

    @model_validator(mode="after")
    def _order(self) -> "DateRange":
        if self.start > self.end:
            raise ValueError("start must be <= end")
        if (self.end - self.start).days > 366:
            raise ValueError("date range cannot exceed 1 year")
        return self


class TransactionHistoryInput(StrictModel):
    account_id: AccountId
    date_range: DateRange
    limit: int = Field(default=20, ge=1, le=100)


class TransactionItem(StrictModel):
    transaction_id: str
    posted_at: datetime
    description: str
    amount: Decimal
    direction: Literal["debit", "credit"]
    merchant_category: str | None = None


class TransactionHistoryOutput(StrictModel):
    account_id: str
    count: int
    transactions: list[TransactionItem]


class StatementInput(StrictModel):
    account_id: AccountId
    statement_period: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")   # YYYY-MM


class StatementOutput(StrictModel):
    account_id: str
    statement_period: str
    download_url: str
    expires_at: datetime
    opening_balance: Decimal
    closing_balance: Decimal


# ---------------------------------------------------------------- payments --
class InternalTransferInput(StrictModel):
    source_acc: AccountId
    dest_acc: AccountId
    amount: Money

    @model_validator(mode="after")
    def _distinct(self) -> "InternalTransferInput":
        if self.source_acc == self.dest_acc:
            raise ValueError("source and destination accounts must differ")
        return self


class TransferOutput(StrictModel):
    transfer_id: str
    status: Literal["completed", "pending"]
    source_acc: str
    dest_acc: str
    amount: Decimal
    new_source_balance: Decimal
    idempotent_replay: bool = False


class BillPayInput(StrictModel):
    payee_id: PayeeId
    amount: Money
    date: date

    @model_validator(mode="after")
    def _future(self) -> "BillPayInput":
        if self.date < date.today():
            raise ValueError("payment date cannot be in the past")
        return self


class BillPayOutput(StrictModel):
    payment_id: str
    payee_id: str
    payee_name: str
    amount: Decimal
    scheduled_date: date
    status: Literal["scheduled"]


class RoutingInput(StrictModel):
    routing_num: RoutingNumber


class RoutingOutput(StrictModel):
    routing_num: str
    valid_checksum: bool
    bank_name: str | None = None
    supports_wire: bool = False


class ExchangeRateInput(StrictModel):
    base_currency: CurrencyCode
    target_currency: CurrencyCode


class ExchangeRateOutput(StrictModel):
    base_currency: str
    target_currency: str
    rate: Decimal
    as_of: datetime
    source: str


# ------------------------------------------------------------------- cards --
class FreezeInput(StrictModel):
    card_id: CardId
    status: Literal["freeze", "unfreeze"]


class CardStatusOutput(StrictModel):
    card_id: str
    masked_pan: str
    status: Literal["active", "frozen", "blocked_lost", "blocked_stolen", "replaced"]
    message: str


class LostStolenInput(StrictModel):
    card_id: CardId


class ReplacementInput(StrictModel):
    card_id: CardId
    delivery_address: str = Field(min_length=10, max_length=300)


class ReplacementOutput(StrictModel):
    order_id: str
    old_card_id: str
    new_card_masked_pan: str
    estimated_delivery: date


class AtmLimitInput(StrictModel):
    card_id: CardId
    new_limit: Money


class AtmLimitOutput(StrictModel):
    card_id: str
    previous_limit: Decimal
    new_limit: Decimal


# ------------------------------------------------------------------- fraud --
class FlagFraudInput(StrictModel):
    transaction_id: TransactionId
    reason: str = Field(min_length=5, max_length=500)


class FlagFraudOutput(StrictModel):
    case_id: str
    transaction_id: str
    status: Literal["under_investigation"]
    provisional_credit_eligible: bool


class UserInput(StrictModel):
    user_id: UserId


class LockProfileOutput(StrictModel):
    user_id: str
    locked: bool
    sessions_revoked: int


class MfaInput(StrictModel):
    user_id: UserId
    channel_type: Literal["sms", "email", "push", "voice"]


class MfaOutput(StrictModel):
    challenge_id: str
    channel_type: str
    expires_in_s: int


class SessionItem(StrictModel):
    session_id: str
    device: str
    ip_masked: str
    location: str
    last_seen: datetime


class ActiveSessionsOutput(StrictModel):
    user_id: str
    sessions: list[SessionItem]


# ------------------------------------------------------------------- loans --
class LoanRatesInput(StrictModel):
    product_type: Literal["mortgage_30y_fixed", "mortgage_15y_fixed", "mortgage_5_1_arm",
                          "auto_new", "auto_used", "personal", "heloc"]


class LoanRatesOutput(StrictModel):
    product_type: str
    apr_from: Decimal
    apr_to: Decimal
    term_months_options: list[int]
    as_of: date
    disclosure: str


class AmortizationInput(StrictModel):
    principal: Money
    interest_rate: Decimal = Field(ge=0, le=40)          # annual %, e.g. 6.5
    term_months: int = Field(ge=1, le=480)


class AmortizationRow(StrictModel):
    month: int
    payment: Decimal
    principal: Decimal
    interest: Decimal
    balance: Decimal


class AmortizationOutput(StrictModel):
    monthly_payment: Decimal
    total_interest: Decimal
    total_paid: Decimal
    schedule_preview: list[AmortizationRow]   # first 12 rows + final row
    term_months: int


class LoanStatusInput(StrictModel):
    application_id: LoanApplicationId


class LoanStatusOutput(StrictModel):
    application_id: str
    product_type: str
    status: Literal["received", "underwriting", "conditionally_approved", "approved", "declined", "funded"]
    last_updated: datetime
    next_step: str


# ------------------------------------------------------------- investments --
class CdRate(StrictModel):
    term_months: int
    apy: Decimal
    minimum_deposit: Decimal


class CdRatesOutput(StrictModel):
    rates: list[CdRate]
    as_of: date
    disclosure: str


class PortfolioInput(StrictModel):
    investment_acc_id: InvestmentAccountId


class Holding(StrictModel):
    fund_id: str
    name: str
    units: Decimal
    nav: Decimal
    market_value: Decimal


class PortfolioOutput(StrictModel):
    investment_acc_id: str
    account_type: str
    total_value: Decimal
    holdings: list[Holding]
    as_of: datetime


class FundSubscriptionInput(StrictModel):
    acc_id: InvestmentAccountId
    fund_id: FundId
    amount: Money


class FundSubscriptionOutput(StrictModel):
    order_id: str
    fund_id: str
    amount: Decimal
    status: Literal["accepted"]
    settlement_date: date


# --------------------------------------------------------------- knowledge --
class KBQueryInput(StrictModel):
    semantic_search_string: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=4, ge=1, le=10)


class KBChunk(StrictModel):
    doc_id: str
    title: str
    content: str
    score: float
    source: str


class KBQueryOutput(StrictModel):
    query: str
    chunks: list[KBChunk]


class LocateInput(StrictModel):
    zip_code_or_coords: str = Field(min_length=3, max_length=60)


class BranchItem(StrictModel):
    branch_id: str
    name: str
    kind: Literal["branch", "atm"]
    address: str
    distance_km: float
    hours: str


class LocateOutput(StrictModel):
    origin: str
    results: list[BranchItem]


# -------------------------------------------------------------- compliance --
class RedactInput(StrictModel):
    text_payload: str = Field(min_length=0, max_length=20000)


class RedactOutput(StrictModel):
    redacted_text: str
    entities_found: dict[str, int]


class AuditInput(StrictModel):
    agent_response_text: str = Field(min_length=1, max_length=20000)


class AuditFinding(StrictModel):
    rule: str
    regulation: str
    severity: Literal["low", "medium", "high"]
    excerpt: str


class AuditOutput(StrictModel):
    compliant: bool
    findings: list[AuditFinding]
    required_disclosures: list[str]


class EmptyInput(StrictModel):
    pass


# Registry used by sub-agents to canonicalise LLM-produced arguments (so approval hashes match
# exactly what the MCP server validates) and to give the LLM precise validation feedback.
TOOL_INPUT_MODELS: dict[str, type[StrictModel]] = {
    "get_account_balance": GetBalanceInput,
    "fetch_transaction_history": TransactionHistoryInput,
    "download_monthly_statement": StatementInput,
    "initiate_internal_transfer": InternalTransferInput,
    "schedule_bill_pay": BillPayInput,
    "verify_routing_number": RoutingInput,
    "get_exchange_rates": ExchangeRateInput,
    "freeze_or_unfreeze_card": FreezeInput,
    "report_lost_stolen_card": LostStolenInput,
    "order_replacement_card": ReplacementInput,
    "update_atm_withdrawal_limit": AtmLimitInput,
    "flag_fraudulent_transaction": FlagFraudInput,
    "lock_user_profile": UserInput,
    "trigger_mfa_challenge": MfaInput,
    "fetch_active_sessions": UserInput,
    "get_loan_product_rates": LoanRatesInput,
    "calculate_amortization_schedule": AmortizationInput,
    "fetch_loan_application_status": LoanStatusInput,
    "fetch_cd_maturation_rates": EmptyInput,
    "get_portfolio_valuation": PortfolioInput,
    "execute_fund_subscription": FundSubscriptionInput,
    "query_knowledge_base": KBQueryInput,
    "locate_nearest_atm_or_branch": LocateInput,
    "redact_pii": RedactInput,
    "audit_compliance_check": AuditInput,
}

# Registry used by the MCP client to verify tool outputs at the boundary.
TOOL_OUTPUT_MODELS: dict[str, type[StrictModel]] = {
    "get_account_balance": BalanceOutput,
    "fetch_transaction_history": TransactionHistoryOutput,
    "download_monthly_statement": StatementOutput,
    "initiate_internal_transfer": TransferOutput,
    "schedule_bill_pay": BillPayOutput,
    "verify_routing_number": RoutingOutput,
    "get_exchange_rates": ExchangeRateOutput,
    "freeze_or_unfreeze_card": CardStatusOutput,
    "report_lost_stolen_card": CardStatusOutput,
    "order_replacement_card": ReplacementOutput,
    "update_atm_withdrawal_limit": AtmLimitOutput,
    "flag_fraudulent_transaction": FlagFraudOutput,
    "lock_user_profile": LockProfileOutput,
    "trigger_mfa_challenge": MfaOutput,
    "fetch_active_sessions": ActiveSessionsOutput,
    "get_loan_product_rates": LoanRatesOutput,
    "calculate_amortization_schedule": AmortizationOutput,
    "fetch_loan_application_status": LoanStatusOutput,
    "fetch_cd_maturation_rates": CdRatesOutput,
    "get_portfolio_valuation": PortfolioOutput,
    "execute_fund_subscription": FundSubscriptionOutput,
    "query_knowledge_base": KBQueryOutput,
    "locate_nearest_atm_or_branch": LocateOutput,
    "redact_pii": RedactOutput,
    "audit_compliance_check": AuditOutput,
}
