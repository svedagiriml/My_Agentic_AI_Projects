"""MCP server: Payments & Money Transfer (HIGH RISK — HITL enforced server-side)."""
from __future__ import annotations

from decimal import Decimal

from bankcore.core_banking.gateway import MockCoreBanking, get_gateway
from bankcore.mcp.server import BankMCPServer, ToolContext
from bankcore.schemas import tools as T


def create(gateway: MockCoreBanking | None = None, **kwargs) -> BankMCPServer:
    gw = gateway or get_gateway()
    srv = BankMCPServer(
        "payments", "Money movement for the authenticated customer. Amounts at/above the approval threshold "
                    "require a bank-officer approval token bound to the exact arguments.",
        tool_rate_limits={"initiate_internal_transfer": (5, 3600), "schedule_bill_pay": (10, 3600)}, **kwargs)

    @srv.tool(name="initiate_internal_transfer", input_model=T.InternalTransferInput, output_model=T.TransferOutput,
              high_risk=True,
              description="Transfer money between two of the customer's OWN accounts. "
                          "source_acc/dest_acc format ACC-######, amount in USD with max 2 decimals.")
    async def initiate_internal_transfer(ctx: ToolContext, source_acc: str, dest_acc: str, amount: float) -> dict:
        return await gw.transfer(ctx.user_id, source_acc, dest_acc, Decimal(str(amount)), ctx.idempotency_key)

    @srv.tool(name="schedule_bill_pay", input_model=T.BillPayInput, output_model=T.BillPayOutput, high_risk=True,
              description="Schedule a bill payment to a saved payee (PAY-####) on a date (YYYY-MM-DD, today or later).")
    async def schedule_bill_pay(ctx: ToolContext, payee_id: str, amount: float, date: str) -> dict:
        return await gw.bill_pay(ctx.user_id, payee_id, Decimal(str(amount)), date, ctx.idempotency_key)

    @srv.tool(name="verify_routing_number", input_model=T.RoutingInput, output_model=T.RoutingOutput,
              description="Validate a 9-digit ABA routing number (checksum + bank directory + wire support).")
    async def verify_routing_number(ctx: ToolContext, routing_num: str) -> dict:
        return await gw.routing(routing_num)

    @srv.tool(name="get_exchange_rates", input_model=T.ExchangeRateInput, output_model=T.ExchangeRateOutput,
              description="Indicative mid-market FX rate between two ISO-4217 currencies (e.g. USD -> INR).")
    async def get_exchange_rates(ctx: ToolContext, base_currency: str, target_currency: str) -> dict:
        return await gw.fx(base_currency, target_currency)

    return srv
