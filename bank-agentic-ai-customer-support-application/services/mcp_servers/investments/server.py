"""MCP server: Investment & Wealth Management."""
from __future__ import annotations

from decimal import Decimal

from bankcore.core_banking.gateway import MockCoreBanking, get_gateway
from bankcore.mcp.server import BankMCPServer, ToolContext
from bankcore.schemas import tools as T


def create(gateway: MockCoreBanking | None = None, **kwargs) -> BankMCPServer:
    gw = gateway or get_gateway()
    srv = BankMCPServer("investments", "CDs, mutual funds and retirement accounts.",
                        tool_rate_limits={"execute_fund_subscription": (5, 86400)}, **kwargs)

    @srv.tool(name="fetch_cd_maturation_rates", input_model=None, output_model=T.CdRatesOutput,
              description="Current certificate of deposit (CD) APYs by term with minimum deposits.")
    async def fetch_cd_maturation_rates(ctx: ToolContext) -> dict:
        return await gw.cd_rates()

    @srv.tool(name="get_portfolio_valuation", input_model=T.PortfolioInput, output_model=T.PortfolioOutput,
              description="Holdings and market value of the customer's investment/IRA account (INV-######).")
    async def get_portfolio_valuation(ctx: ToolContext, investment_acc_id: str) -> dict:
        return await gw.portfolio(ctx.user_id, investment_acc_id)

    @srv.tool(name="execute_fund_subscription", input_model=T.FundSubscriptionInput,
              output_model=T.FundSubscriptionOutput, high_risk=True,
              description="Buy units of a mutual fund (FND-...) in an investment account (INV-######).")
    async def execute_fund_subscription(ctx: ToolContext, acc_id: str, fund_id: str, amount: float) -> dict:
        return await gw.subscribe_fund(ctx.user_id, acc_id, fund_id, Decimal(str(amount)), ctx.idempotency_key)

    return srv
