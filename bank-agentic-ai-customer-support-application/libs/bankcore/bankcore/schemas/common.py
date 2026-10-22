from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints


class StrictModel(BaseModel):
    """Base for every inter-service message: unknown fields are rejected, strings stripped."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=False,
                              validate_assignment=True)


class AgentName(str, Enum):
    ACCOUNT = "account"
    PAYMENTS = "payments"
    CARDS = "cards"
    FRAUD = "fraud"
    LOANS = "loans"
    INVESTMENTS = "investments"
    KNOWLEDGE = "knowledge"
    COMPLIANCE = "compliance"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Role(str, Enum):
    CUSTOMER = "customer"
    STAFF = "staff"          # human-in-the-loop reviewers (bank officers)
    SERVICE = "service"      # machine identities (Cloud Run service accounts)


# ---- reusable constrained types --------------------------------------------
UserId = Annotated[str, StringConstraints(pattern=r"^USR-\d{4,8}$")]
AccountId = Annotated[str, StringConstraints(pattern=r"^ACC-\d{6}$")]
CardId = Annotated[str, StringConstraints(pattern=r"^CRD-\d{6}$")]
TransactionId = Annotated[str, StringConstraints(pattern=r"^TXN-[A-Z0-9]{6,20}$")]
PayeeId = Annotated[str, StringConstraints(pattern=r"^PAY-\d{4,8}$")]
LoanApplicationId = Annotated[str, StringConstraints(pattern=r"^APP-\d{6}$")]
InvestmentAccountId = Annotated[str, StringConstraints(pattern=r"^INV-\d{6}$")]
FundId = Annotated[str, StringConstraints(pattern=r"^FND-[A-Z0-9]{3,10}$")]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
RoutingNumber = Annotated[str, StringConstraints(pattern=r"^\d{9}$")]
def _quantize_money(v: Decimal) -> Decimal:
    # Canonical 2-dp form so "250", 250.0 and "250.00" hash identically (approval tokens are
    # bound to the canonical JSON of the arguments).
    return v.quantize(Decimal("0.01"))


Money = Annotated[Decimal, Field(gt=0, le=Decimal("1000000"), decimal_places=2), AfterValidator(_quantize_money)]
SafeText = Annotated[str, StringConstraints(min_length=1, max_length=4000)]


class TokenUsage(StrictModel):
    model: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(model=self.model if self.model != "unknown" else other.model,
                          input_tokens=self.input_tokens + other.input_tokens,
                          output_tokens=self.output_tokens + other.output_tokens)


class ErrorInfo(StrictModel):
    code: str
    message: str
    retryable: bool = False
