"""MCP server: Card Operations."""
from __future__ import annotations

from decimal import Decimal

from bankcore.core_banking.gateway import MockCoreBanking, get_gateway
from bankcore.mcp.server import BankMCPServer, ToolContext
from bankcore.schemas import tools as T


def create(gateway: MockCoreBanking | None = None, **kwargs) -> BankMCPServer:
    gw = gateway or get_gateway()
    srv = BankMCPServer("cards", "Card controls for the authenticated customer's cards (CRD-######).",
                        tool_rate_limits={"order_replacement_card": (3, 86400),
                                          "update_atm_withdrawal_limit": (5, 86400)}, **kwargs)

    @srv.tool(name="freeze_or_unfreeze_card", input_model=T.FreezeInput, output_model=T.CardStatusOutput,
              description="Temporarily freeze or unfreeze a card. status must be 'freeze' or 'unfreeze'.")
    async def freeze_or_unfreeze_card(ctx: ToolContext, card_id: str, status: str) -> dict:
        return await gw.freeze(ctx.user_id, card_id, status)

    @srv.tool(name="report_lost_stolen_card", input_model=T.LostStolenInput, output_model=T.CardStatusOutput,
              description="Permanently block a lost or stolen card immediately. Cannot be undone.")
    async def report_lost_stolen_card(ctx: ToolContext, card_id: str) -> dict:
        return await gw.report_lost(ctx.user_id, card_id)

    @srv.tool(name="order_replacement_card", input_model=T.ReplacementInput, output_model=T.ReplacementOutput,
              description="Order a replacement for a blocked or frozen card, delivered to delivery_address.")
    async def order_replacement_card(ctx: ToolContext, card_id: str, delivery_address: str) -> dict:
        return await gw.replace(ctx.user_id, card_id, delivery_address)

    @srv.tool(name="update_atm_withdrawal_limit", input_model=T.AtmLimitInput, output_model=T.AtmLimitOutput,
              high_risk=True, description="Change the daily ATM cash withdrawal limit (USD) of a card.")
    async def update_atm_withdrawal_limit(ctx: ToolContext, card_id: str, new_limit: float) -> dict:
        return await gw.atm_limit(ctx.user_id, card_id, Decimal(str(new_limit)))

    return srv
