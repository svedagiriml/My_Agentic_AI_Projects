"""MCP server: Account & Balance Management (read-heavy, low risk)."""
from __future__ import annotations

from bankcore.core_banking.gateway import MockCoreBanking, get_gateway
from bankcore.mcp.server import BankMCPServer, ToolContext
from bankcore.schemas import tools as T


def create(gateway: MockCoreBanking | None = None, **kwargs) -> BankMCPServer:
    gw = gateway or get_gateway()
    srv = BankMCPServer("account", "Read-only account data for the authenticated customer.",
                        tool_rate_limits={"download_monthly_statement": (10, 3600)}, **kwargs)

    @srv.tool(name="get_account_balance", input_model=T.GetBalanceInput, output_model=T.BalanceOutput,
              description="Get the available and ledger balance of one of the customer's accounts "
                          "(checking, savings or credit). account_id format: ACC-######.")
    async def get_account_balance(ctx: ToolContext, account_id: str) -> dict:
        return await gw.balance(ctx.user_id, account_id)

    @srv.tool(name="fetch_transaction_history", input_model=T.TransactionHistoryInput,
              output_model=T.TransactionHistoryOutput,
              description="List posted transactions for an account within date_range "
                          "{start: YYYY-MM-DD, end: YYYY-MM-DD} (max 1 year), newest first, up to `limit` (1-100).")
    async def fetch_transaction_history(ctx: ToolContext, account_id: str, date_range: dict, limit: int = 20) -> dict:
        return await gw.transactions(ctx.user_id, account_id, date_range["start"], date_range["end"], limit)

    @srv.tool(name="download_monthly_statement", input_model=T.StatementInput, output_model=T.StatementOutput,
              description="Generate a short-lived secure download link for a monthly statement. "
                          "statement_period format: YYYY-MM.")
    async def download_monthly_statement(ctx: ToolContext, account_id: str, statement_period: str) -> dict:
        return await gw.statement(ctx.user_id, account_id, statement_period)

    return srv
