"""MCP tool outputs, state isolation, ownership, HITL enforcement and core-banking rules."""
from __future__ import annotations

from decimal import Decimal

import pytest

from bankcore.core_banking.gateway import MockCoreBanking, aba_checksum_ok
from bankcore.errors import ToolNotPermitted
from bankcore.mcp.client import MCPToolClient
from bankcore.schemas.common import RiskLevel
from bankcore.schemas.messages import ProposedAction
from bankcore.schemas.tools import TOOL_OUTPUT_MODELS
from bankcore.security.signing import issue_approval_token, issue_delegation_token

from mcp_servers.accounts.server import create as accounts
from mcp_servers.cards.server import create as cards
from mcp_servers.compliance.rules import audit
from mcp_servers.knowledge.server import parse_origin
from mcp_servers.payments.server import create as payments


def _client(srv, agent, tools):
    return MCPToolClient(agent=agent, domain=agent, target=srv.server, allowed_tools=tools)


def _tok(agent, user="USR-1001"):
    return issue_delegation_token(user_id=user, agent=agent, thread_id="th-test")


async def test_account_tools_validated_output():
    srv = accounts(gateway=MockCoreBanking())
    async with _client(srv, "account", {"get_account_balance", "fetch_transaction_history"}).session() as s:
        env, _ = await s.call("get_account_balance", {"account_id": "ACC-100001"}, delegation_token=_tok("account"))
        assert env.ok and TOOL_OUTPUT_MODELS["get_account_balance"].model_validate(env.data)
        assert env.data["available_balance"] == "5240.75"
        env, _ = await s.call("fetch_transaction_history", {"account_id": "ACC-100001", "limit": 3,
                                                            "date_range": {"start": "2026-09-01", "end": "2026-10-19"}},
                              delegation_token=_tok("account"))
        assert env.ok and env.data["count"] == 3


async def test_ownership_enforced_from_token_not_arguments():
    srv = accounts(gateway=MockCoreBanking())
    async with _client(srv, "account", {"get_account_balance"}).session() as s:
        env, _ = await s.call("get_account_balance", {"account_id": "ACC-100001"},
                              delegation_token=_tok("account", user="USR-1002"))
        assert not env.ok and env.error.code == "not_found"     # no enumeration of other customers' accounts


async def test_state_isolation_allow_list_and_token_audience():
    srv = payments(gateway=MockCoreBanking())
    c = _client(srv, "payments", {"get_exchange_rates"})
    async with c.session() as s:
        names = [t["function"]["name"] for t in await s.llm_tools()]
        assert names == ["get_exchange_rates"]
        assert all("delegation_token" not in t["function"]["parameters"].get("properties", {})
                   for t in await s.llm_tools())
        with pytest.raises(ToolNotPermitted):
            await s.call("report_lost_stolen_card", {"card_id": "CRD-100001"}, delegation_token=_tok("payments"))
        env, _ = await s.call("get_exchange_rates", {"base_currency": "USD", "target_currency": "EUR"},
                              delegation_token=_tok("cards"))         # token minted for another domain
        assert not env.ok and env.error.code == "forbidden"


async def test_hitl_enforced_server_side_and_single_use():
    srv = payments(gateway=MockCoreBanking())
    args = {"source_acc": "ACC-100002", "dest_acc": "ACC-100001", "amount": "5000.00"}
    async with _client(srv, "payments", {"initiate_internal_transfer"}).session() as s:
        env, _ = await s.call("initiate_internal_transfer", args, delegation_token=_tok("payments"))
        assert env.requires_approval and env.error.code == "approval_required"
        action = ProposedAction(agent="payments", tool="initiate_internal_transfer", arguments=args,
                                risk_level=RiskLevel.HIGH, reason="r", summary_for_reviewer="s")
        appr = issue_approval_token(action, user_id="USR-1001", reviewer_id="STF-001")
        bad, _ = await s.call("initiate_internal_transfer", {**args, "amount": "9000.00"},
                              delegation_token=_tok("payments"), approval_token=appr)
        assert not bad.ok
        ok, _ = await s.call("initiate_internal_transfer", args, delegation_token=_tok("payments"), approval_token=appr)
        assert ok.ok and ok.data["amount"] == "5000.00"
        again, _ = await s.call("initiate_internal_transfer", args, delegation_token=_tok("payments"),
                                approval_token=appr)
        assert not again.ok                                            # replay of approval token refused


async def test_small_transfer_idempotent_and_business_rules():
    gw = MockCoreBanking()
    srv = payments(gateway=gw)
    args = {"source_acc": "ACC-100001", "dest_acc": "ACC-100002", "amount": 100}
    async with _client(srv, "payments", {"initiate_internal_transfer"}).session() as s:
        first, _ = await s.call("initiate_internal_transfer", args, delegation_token=_tok("payments"))
        second, _ = await s.call("initiate_internal_transfer", args, delegation_token=_tok("payments"))
        assert first.data["transfer_id"] == second.data["transfer_id"] and second.data["idempotent_replay"]
        assert gw.s.accounts["ACC-100001"].balance == Decimal("5140.75")      # debited once
        bad, _ = await s.call("initiate_internal_transfer", {**args, "amount": 999}, delegation_token=_tok("payments"))
        assert bad.ok or bad.error.code != "internal_error"
        invalid, _ = await s.call("initiate_internal_transfer", {**args, "amount": -5}, delegation_token=_tok("payments"))
        assert invalid.error.code == "invalid_arguments"


async def test_card_lost_then_replacement():
    srv = cards(gateway=MockCoreBanking())
    tok = _tok("cards")
    async with _client(srv, "cards", {"report_lost_stolen_card", "order_replacement_card"}).session() as s:
        env, _ = await s.call("report_lost_stolen_card", {"card_id": "CRD-100001"}, delegation_token=tok)
        assert env.data["status"] == "blocked_lost" and env.data["masked_pan"].endswith("4821")
        env, _ = await s.call("order_replacement_card", {"card_id": "CRD-100001",
                                                         "delivery_address": "12 Elm Street, Springfield, IL 62701"},
                              delegation_token=tok)
        assert env.ok and env.data["old_card_id"] == "CRD-100001"


def test_amortization_math():
    out = MockCoreBanking.amortization(Decimal("300000"), Decimal("6.5"), 360)
    assert out["monthly_payment"] == Decimal("1896.20")
    assert out["schedule_preview"][-1]["balance"] == Decimal("0.00")


def test_aba_checksum_and_location_parsing():
    assert aba_checksum_ok("071000990") and aba_checksum_ok("021000021") and not aba_checksum_ok("123456789")
    assert parse_origin("62701") == (39.8017, -89.6436)
    assert parse_origin("39.80, -89.64") == (39.80, -89.64)


def test_compliance_rules():
    bad = audit("This fund offers guaranteed returns and you will definitely get approved for the loan.")
    assert not bad["compliant"] and {f["rule"] for f in bad["findings"]} >= {"no_guaranteed_returns",
                                                                              "no_guaranteed_credit"}
    ok = audit("Our 12-month CD earns 4.50% APY.")
    assert ok["compliant"] and ok["required_disclosures"]
    pii = audit("Your card 4111 1111 1111 1111 is active")
    assert not pii["compliant"]
