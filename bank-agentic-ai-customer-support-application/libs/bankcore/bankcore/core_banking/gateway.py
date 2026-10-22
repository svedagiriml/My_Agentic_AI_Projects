"""Core-banking gateway.

MCP servers never talk to databases directly; they call this gateway, which in production is
an adapter over the bank's system of record (Temenos / Finacle / FIS / Fiserv REST APIs,
mainframe via MQ, etc.). ``MockCoreBanking`` is a realistic, deterministic, in-memory
implementation with the same business rules (ownership, limits, idempotency, balances),
used for local development, demos, evaluation and tests.

Ownership is enforced HERE using the user id from the verified delegation token — never from
LLM-produced arguments. Unknown and not-owned resources both return NotFound (no enumeration).
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

from bankcore.errors import BankingDomainError, NotFound

BANK_NAME = "Horizon Trust Bank"
BANK_ROUTING = "071000990"
TWOPLACES = Decimal("0.01")


def q(x: Decimal | float | str) -> Decimal:
    return Decimal(str(x)).quantize(TWOPLACES, rounding=ROUND_HALF_UP)


def aba_checksum_ok(routing: str) -> bool:
    if len(routing) != 9 or not routing.isdigit():
        return False
    d = [int(c) for c in routing]
    return (3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + (d[2] + d[5] + d[8])) % 10 == 0


@dataclass
class Account:
    account_id: str
    user_id: str
    account_type: str
    balance: Decimal
    currency: str = "USD"
    credit_limit: Decimal | None = None
    holds: Decimal = Decimal("0")


@dataclass
class Card:
    card_id: str
    user_id: str
    account_id: str
    kind: str
    pan_last4: str
    status: str = "active"
    atm_limit: Decimal = Decimal("500.00")


@dataclass
class Txn:
    transaction_id: str
    account_id: str
    posted_at: datetime
    description: str
    amount: Decimal
    direction: str
    mcc: str | None = None


@dataclass
class MockState:
    users: dict[str, dict] = field(default_factory=dict)
    accounts: dict[str, Account] = field(default_factory=dict)
    cards: dict[str, Card] = field(default_factory=dict)
    txns: list[Txn] = field(default_factory=list)
    payees: dict[str, dict] = field(default_factory=dict)
    loan_apps: dict[str, dict] = field(default_factory=dict)
    investments: dict[str, dict] = field(default_factory=dict)
    sessions: dict[str, list[dict]] = field(default_factory=dict)
    idempotency: dict[str, dict] = field(default_factory=dict)
    fraud_cases: dict[str, dict] = field(default_factory=dict)
    daily_transfer_total: dict[tuple[str, date], Decimal] = field(default_factory=dict)


FUNDS = {
    "FND-SPX500": {"name": "Horizon S&P 500 Index Fund", "nav": Decimal("512.34"), "min": Decimal("100")},
    "FND-BOND": {"name": "Horizon Aggregate Bond Fund", "nav": Decimal("98.12"), "min": Decimal("100")},
    "FND-INTL": {"name": "Horizon International Equity Fund", "nav": Decimal("41.77"), "min": Decimal("250")},
}
ROUTING_DIRECTORY = {
    BANK_ROUTING: (BANK_NAME, True), "021000021": ("JPMorgan Chase Bank", True),
    "026009593": ("Bank of America", True), "121000248": ("Wells Fargo Bank", True),
    "011000015": ("Federal Reserve Bank of Boston", False),
}
FX_MID = {("USD", "EUR"): "0.9210", ("USD", "GBP"): "0.7740", ("USD", "INR"): "84.1500", ("USD", "AUD"): "1.5120",
          ("USD", "JPY"): "149.8000", ("USD", "CAD"): "1.3780", ("USD", "SGD"): "1.3350"}
LOAN_RATES = {
    "mortgage_30y_fixed": ("6.375", "7.125", [360]), "mortgage_15y_fixed": ("5.625", "6.250", [180]),
    "mortgage_5_1_arm": ("5.875", "6.750", [360]), "auto_new": ("5.490", "8.990", [36, 48, 60, 72]),
    "auto_used": ("6.240", "10.490", [36, 48, 60]), "personal": ("8.990", "19.990", [24, 36, 48, 60]),
    "heloc": ("7.750", "9.500", [120, 240]),
}
CD_RATES = [(3, "4.10", "500"), (6, "4.35", "500"), (12, "4.50", "1000"), (18, "4.25", "1000"),
            (24, "4.00", "1000"), (60, "3.75", "1000")]


def _seed() -> MockState:
    rnd = random.Random(42)
    st = MockState()
    st.users = {"USR-1001": {"name": "Alex Morgan", "locked": False, "address": "12 Elm Street, Springfield, IL 62701"},
                "USR-1002": {"name": "Jordan Lee", "locked": False, "address": "88 Bay Road, Austin, TX 73301"}}
    st.accounts = {
        "ACC-100001": Account("ACC-100001", "USR-1001", "checking", q("5240.75")),
        "ACC-100002": Account("ACC-100002", "USR-1001", "savings", q("18900.00")),
        "ACC-100003": Account("ACC-100003", "USR-1001", "credit", q("-1250.40"), credit_limit=q("10000")),
        "ACC-200001": Account("ACC-200001", "USR-1002", "checking", q("830.10")),
    }
    st.cards = {
        "CRD-100001": Card("CRD-100001", "USR-1001", "ACC-100001", "debit", "4821"),
        "CRD-100002": Card("CRD-100002", "USR-1001", "ACC-100003", "credit", "7730", atm_limit=q("300")),
        "CRD-200001": Card("CRD-200001", "USR-1002", "ACC-200001", "debit", "1109"),
    }
    merchants = [("Whole Foods Market", "5411"), ("Shell Fuel", "5541"), ("Netflix", "4899"), ("Amazon", "5942"),
                 ("Starbucks", "5814"), ("Uber", "4121"), ("CVS Pharmacy", "5912"), ("Delta Air Lines", "4511")]
    now = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    for acc in ("ACC-100001", "ACC-100003", "ACC-200001"):
        for i in range(60):
            m, mcc = rnd.choice(merchants)
            ts = now - timedelta(days=i * 1.5, hours=rnd.randint(0, 10))
            tid = "TXN-" + hashlib.sha1(f"{acc}{i}".encode()).hexdigest()[:10].upper()
            st.txns.append(Txn(tid, acc, ts, m, q(rnd.uniform(4, 180)), "debit", mcc))
        st.txns.append(Txn("TXN-PAYROLL" + acc[-2:], acc, now - timedelta(days=4), "ACME Corp Payroll",
                           q("3200.00"), "credit", None))
    # A suspicious transaction used by fraud scenarios
    st.txns.append(Txn("TXN-SUSP0001", "ACC-100001", now - timedelta(hours=6), "ELECTRO WORLD LAGOS NG",
                       q("1899.99"), "debit", "5732"))
    st.payees = {"PAY-5001": {"user_id": "USR-1001", "name": "City Power & Light"},
                 "PAY-5002": {"user_id": "USR-1001", "name": "Metro Water Utility"},
                 "PAY-6001": {"user_id": "USR-1002", "name": "Austin Energy"}}
    st.loan_apps = {"APP-300001": {"user_id": "USR-1001", "product_type": "mortgage_30y_fixed",
                                   "status": "underwriting", "updated": now - timedelta(days=2),
                                   "next_step": "Upload your two most recent pay stubs in the document centre."}}
    st.investments = {"INV-400001": {"user_id": "USR-1001", "type": "Traditional IRA",
                                     "holdings": {"FND-SPX500": Decimal("21.5"), "FND-BOND": Decimal("140")}}}
    st.sessions = {"USR-1001": [
        {"session_id": "ses-a1", "device": "iPhone 15 / iOS app", "ip": "73.22.14.9", "location": "Springfield, US",
         "last_seen": now - timedelta(minutes=5)},
        {"session_id": "ses-b7", "device": "Chrome / Windows", "ip": "102.89.3.77", "location": "Lagos, NG",
         "last_seen": now - timedelta(hours=6)}],
        "USR-1002": [{"session_id": "ses-c3", "device": "Pixel 8 / Android app", "ip": "24.5.9.1",
                      "location": "Austin, US", "last_seen": now - timedelta(hours=1)}]}
    return st


class MockCoreBanking:
    def __init__(self) -> None:
        self.s = _seed()
        self._lock = asyncio.Lock()
        self.latency_s = 0.0          # chaos testing knob

    async def _io(self) -> None:
        if self.latency_s:
            await asyncio.sleep(self.latency_s)

    # ------------------------------------------------------------ helpers --
    def _acct(self, user_id: str, account_id: str) -> Account:
        acc = self.s.accounts.get(account_id)
        if acc is None or acc.user_id != user_id:
            raise NotFound(f"account {account_id} not found for user", user_message="Account not found.")
        return acc

    def _card(self, user_id: str, card_id: str) -> Card:
        card = self.s.cards.get(card_id)
        if card is None or card.user_id != user_id:
            raise NotFound(f"card {card_id} not found", user_message="Card not found.")
        return card

    def _ensure_active_user(self, user_id: str) -> None:
        if self.s.users.get(user_id, {}).get("locked"):
            raise BankingDomainError("profile locked", code="profile_locked",
                                     user_message="Your profile is locked for security. Please contact us.")

    # ----------------------------------------------------------- accounts --
    async def balance(self, user_id: str, account_id: str) -> dict:
        await self._io()
        a = self._acct(user_id, account_id)
        available = a.balance - a.holds if a.account_type != "credit" else (a.credit_limit or 0) + a.balance
        return {"account_id": a.account_id, "account_type": a.account_type, "currency": a.currency,
                "available_balance": q(available), "ledger_balance": q(a.balance), "credit_limit": a.credit_limit,
                "as_of": datetime.now(timezone.utc)}

    async def transactions(self, user_id: str, account_id: str, start: date, end: date, limit: int) -> dict:
        await self._io()
        self._acct(user_id, account_id)
        items = [t for t in self.s.txns if t.account_id == account_id and start <= t.posted_at.date() <= end]
        items.sort(key=lambda t: t.posted_at, reverse=True)
        items = items[:limit]
        return {"account_id": account_id, "count": len(items), "transactions": [
            {"transaction_id": t.transaction_id, "posted_at": t.posted_at, "description": t.description,
             "amount": t.amount, "direction": t.direction, "merchant_category": t.mcc} for t in items]}

    async def statement(self, user_id: str, account_id: str, period: str) -> dict:
        await self._io()
        a = self._acct(user_id, account_id)
        y, m = map(int, period.split("-"))
        if date(y, m, 1) >= date.today().replace(day=1):
            raise BankingDomainError("statement not yet generated", code="statement_unavailable",
                                     user_message="That statement is not available yet.")
        token = hashlib.sha256(f"{account_id}{period}{uuid.uuid4()}".encode()).hexdigest()[:24]
        return {"account_id": account_id, "statement_period": period,
                "download_url": f"https://secure.horizontrust.example/statements/{token}.pdf",
                "expires_at": datetime.now(timezone.utc) + timedelta(minutes=15),
                "opening_balance": q(a.balance - Decimal("412.18")), "closing_balance": q(a.balance)}

    # ----------------------------------------------------------- payments --
    async def transfer(self, user_id: str, src: str, dst: str, amount: Decimal, idem_key: str) -> dict:
        async with self._lock:              # serialise money movement (in prod: DB row locks)
            await self._io()
            if idem_key in self.s.idempotency:
                return {**self.s.idempotency[idem_key], "idempotent_replay": True}
            self._ensure_active_user(user_id)
            s, d = self._acct(user_id, src), self._acct(user_id, dst)
            if s.account_type == "credit":
                raise BankingDomainError("cash advance via transfer not allowed", code="unsupported_source",
                                         user_message="Transfers from a credit card account aren't supported.")
            if s.balance - s.holds < amount:
                raise BankingDomainError("insufficient funds", code="insufficient_funds",
                                         user_message="There are insufficient available funds for this transfer.")
            key = (user_id, date.today())
            total = self.s.daily_transfer_total.get(key, Decimal("0")) + amount
            if total > Decimal("25000"):
                raise BankingDomainError("daily limit", code="daily_limit_exceeded",
                                         user_message="This transfer would exceed your daily transfer limit.")
            s.balance -= amount
            d.balance += amount
            self.s.daily_transfer_total[key] = total
            tid = "TRF-" + uuid.uuid4().hex[:10].upper()
            res = {"transfer_id": tid, "status": "completed", "source_acc": src, "dest_acc": dst,
                   "amount": q(amount), "new_source_balance": q(s.balance), "idempotent_replay": False}
            self.s.idempotency[idem_key] = res
            return res

    async def bill_pay(self, user_id: str, payee_id: str, amount: Decimal, when: date, idem_key: str) -> dict:
        async with self._lock:
            if idem_key in self.s.idempotency:
                return self.s.idempotency[idem_key]
            self._ensure_active_user(user_id)
            p = self.s.payees.get(payee_id)
            if p is None or p["user_id"] != user_id:
                raise NotFound("payee not found", user_message="Payee not found. Please add the payee first.")
            if when > date.today() + timedelta(days=365):
                raise BankingDomainError("too far", user_message="Payments can be scheduled up to 1 year ahead.")
            res = {"payment_id": "BP-" + uuid.uuid4().hex[:10].upper(), "payee_id": payee_id, "payee_name": p["name"],
                   "amount": q(amount), "scheduled_date": when, "status": "scheduled"}
            self.s.idempotency[idem_key] = res
            return res

    async def routing(self, routing: str) -> dict:
        await self._io()
        name, wire = ROUTING_DIRECTORY.get(routing, (None, False))
        return {"routing_num": routing, "valid_checksum": aba_checksum_ok(routing), "bank_name": name,
                "supports_wire": wire}

    async def fx(self, base: str, target: str) -> dict:
        await self._io()
        if base == target:
            rate = Decimal("1")
        elif (base, target) in FX_MID:
            rate = Decimal(FX_MID[(base, target)])
        elif (target, base) in FX_MID:
            rate = (Decimal("1") / Decimal(FX_MID[(target, base)])).quantize(Decimal("0.000001"))
        elif base != "USD" and target != "USD" and ("USD", base) in FX_MID and ("USD", target) in FX_MID:
            rate = (Decimal(FX_MID[("USD", target)]) / Decimal(FX_MID[("USD", base)])).quantize(Decimal("0.000001"))
        else:
            raise NotFound("currency pair not supported", user_message="That currency pair isn't supported.")
        return {"base_currency": base, "target_currency": target, "rate": rate,
                "as_of": datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
                "source": "Horizon Treasury mid-market (indicative; final rate set at execution)"}

    # -------------------------------------------------------------- cards --
    def _card_out(self, c: Card, message: str) -> dict:
        return {"card_id": c.card_id, "masked_pan": f"**** **** **** {c.pan_last4}", "status": c.status,
                "message": message}

    async def freeze(self, user_id: str, card_id: str, status: str) -> dict:
        async with self._lock:
            c = self._card(user_id, card_id)
            if c.status.startswith("blocked") or c.status == "replaced":
                raise BankingDomainError("card permanently blocked", code="card_blocked",
                                         user_message="This card is permanently blocked; please order a replacement.")
            c.status = "frozen" if status == "freeze" else "active"
            return self._card_out(c, f"Card {'frozen' if status == 'freeze' else 'unfrozen'} successfully.")

    async def report_lost(self, user_id: str, card_id: str) -> dict:
        async with self._lock:
            c = self._card(user_id, card_id)
            c.status = "blocked_lost"
            return self._card_out(c, "Card permanently blocked. Pending authorisations will be declined.")

    async def replace(self, user_id: str, card_id: str, address: str) -> dict:
        async with self._lock:
            c = self._card(user_id, card_id)
            if c.status == "active":
                raise BankingDomainError("card active", code="card_active",
                                         user_message="The card is active; report it lost/stolen or freeze it first.")
            new_last4 = f"{random.Random(card_id).randint(1000, 9999)}"
            c.status = "replaced"
            return {"order_id": "ORD-" + uuid.uuid4().hex[:8].upper(), "old_card_id": card_id,
                    "new_card_masked_pan": f"**** **** **** {new_last4}",
                    "estimated_delivery": date.today() + timedelta(days=5)}

    async def atm_limit(self, user_id: str, card_id: str, new_limit: Decimal) -> dict:
        async with self._lock:
            c = self._card(user_id, card_id)
            if new_limit > Decimal("5000"):
                raise BankingDomainError("above product max", code="limit_too_high",
                                         user_message="The maximum daily ATM limit for this card is $5,000.")
            prev, c.atm_limit = c.atm_limit, q(new_limit)
            return {"card_id": card_id, "previous_limit": prev, "new_limit": c.atm_limit}

    # -------------------------------------------------------------- fraud --
    async def flag_fraud(self, user_id: str, txn_id: str, reason: str) -> dict:
        async with self._lock:
            t = next((t for t in self.s.txns if t.transaction_id == txn_id), None)
            if t is None or self.s.accounts[t.account_id].user_id != user_id:
                raise NotFound("txn not found", user_message="Transaction not found.")
            case = {"case_id": "FRD-" + uuid.uuid4().hex[:8].upper(), "transaction_id": txn_id,
                    "status": "under_investigation",
                    "provisional_credit_eligible": t.direction == "debit" and t.amount <= Decimal("5000")}
            self.s.fraud_cases[case["case_id"]] = {**case, "reason": reason}
            return case

    async def lock_profile(self, user_id: str, target: str) -> dict:
        if target != user_id:
            raise NotFound("user not found", user_message="Profile not found.")
        async with self._lock:
            self.s.users[target]["locked"] = True
            revoked = len(self.s.sessions.get(target, []))
            self.s.sessions[target] = []
            return {"user_id": target, "locked": True, "sessions_revoked": revoked}

    async def mfa(self, user_id: str, target: str, channel: str) -> dict:
        if target != user_id:
            raise NotFound("user not found", user_message="Profile not found.")
        return {"challenge_id": "MFA-" + uuid.uuid4().hex[:10].upper(), "channel_type": channel, "expires_in_s": 300}

    async def sessions(self, user_id: str, target: str) -> dict:
        if target != user_id:
            raise NotFound("user not found", user_message="Profile not found.")
        out = []
        for s in self.s.sessions.get(target, []):
            ip = s["ip"].split(".")
            out.append({"session_id": s["session_id"], "device": s["device"], "ip_masked": f"{ip[0]}.{ip[1]}.x.x",
                        "location": s["location"], "last_seen": s["last_seen"]})
        return {"user_id": target, "sessions": out}

    # -------------------------------------------------------------- loans --
    async def loan_rates(self, product: str) -> dict:
        lo, hi, terms = LOAN_RATES[product]
        return {"product_type": product, "apr_from": Decimal(lo), "apr_to": Decimal(hi), "term_months_options": terms,
                "as_of": date.today(),
                "disclosure": "APR ranges are indicative, based on creditworthiness, and subject to change. "
                              "Not a commitment to lend. Equal Housing Lender."}

    @staticmethod
    def amortization(principal: Decimal, annual_rate: Decimal, n: int) -> dict:
        r = float(annual_rate) / 100 / 12
        p = float(principal)
        pay = p / n if r == 0 else p * r / (1 - math.pow(1 + r, -n))
        bal, rows, total_int = p, [], 0.0
        for month in range(1, n + 1):
            interest = bal * r
            princ = pay - interest
            if month == n:
                princ, pay_m = bal, bal + interest
            else:
                pay_m = pay
            bal = max(0.0, bal - princ)
            total_int += interest
            if month <= 12 or month == n:
                rows.append({"month": month, "payment": q(pay_m), "principal": q(princ), "interest": q(interest),
                             "balance": q(bal)})
        return {"monthly_payment": q(pay), "total_interest": q(total_int), "total_paid": q(p + total_int),
                "schedule_preview": rows, "term_months": n}

    async def loan_status(self, user_id: str, app_id: str) -> dict:
        a = self.s.loan_apps.get(app_id)
        if a is None or a["user_id"] != user_id:
            raise NotFound("application not found", user_message="Loan application not found.")
        return {"application_id": app_id, "product_type": a["product_type"], "status": a["status"],
                "last_updated": a["updated"], "next_step": a["next_step"]}

    # -------------------------------------------------------- investments --
    async def cd_rates(self) -> dict:
        return {"rates": [{"term_months": t, "apy": Decimal(a), "minimum_deposit": Decimal(m)} for t, a, m in CD_RATES],
                "as_of": date.today(),
                "disclosure": "APY = Annual Percentage Yield, accurate as of the date shown. Penalty may be imposed "
                              "for early withdrawal. Fees may reduce earnings. (Truth in Savings Act / Reg DD)"}

    async def portfolio(self, user_id: str, inv_id: str) -> dict:
        inv = self.s.investments.get(inv_id)
        if inv is None or inv["user_id"] != user_id:
            raise NotFound("investment account not found", user_message="Investment account not found.")
        holdings, total = [], Decimal("0")
        for fid, units in inv["holdings"].items():
            mv = q(units * FUNDS[fid]["nav"])
            total += mv
            holdings.append({"fund_id": fid, "name": FUNDS[fid]["name"], "units": units, "nav": FUNDS[fid]["nav"],
                             "market_value": mv})
        return {"investment_acc_id": inv_id, "account_type": inv["type"], "total_value": q(total),
                "holdings": holdings, "as_of": datetime.now(timezone.utc)}

    async def subscribe_fund(self, user_id: str, inv_id: str, fund_id: str, amount: Decimal, idem_key: str) -> dict:
        async with self._lock:
            if idem_key in self.s.idempotency:
                return self.s.idempotency[idem_key]
            inv = self.s.investments.get(inv_id)
            if inv is None or inv["user_id"] != user_id:
                raise NotFound("investment account not found", user_message="Investment account not found.")
            f = FUNDS.get(fund_id)
            if f is None:
                raise NotFound("fund not found", user_message="Fund not found.")
            if amount < f["min"]:
                raise BankingDomainError("below minimum", code="below_minimum",
                                         user_message=f"The minimum investment for this fund is ${f['min']}.")
            inv["holdings"][fund_id] = inv["holdings"].get(fund_id, Decimal("0")) + (amount / f["nav"]).quantize(
                Decimal("0.0001"))
            res = {"order_id": "SUB-" + uuid.uuid4().hex[:8].upper(), "fund_id": fund_id, "amount": q(amount),
                   "status": "accepted", "settlement_date": date.today() + timedelta(days=1)}
            self.s.idempotency[idem_key] = res
            return res


_gateway: MockCoreBanking | None = None


def get_gateway() -> MockCoreBanking:
    global _gateway
    if _gateway is None:
        _gateway = MockCoreBanking()
    return _gateway


def reset_gateway() -> None:
    global _gateway
    _gateway = None
