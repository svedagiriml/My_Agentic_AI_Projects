"""MCP server: Loans & Mortgages."""
from __future__ import annotations

from decimal import Decimal

from bankcore.core_banking.gateway import MockCoreBanking, get_gateway
from bankcore.mcp.server import BankMCPServer, ToolContext
from bankcore.schemas import tools as T


def create(gateway: MockCoreBanking | None = None, **kwargs) -> BankMCPServer:
    gw = gateway or get_gateway()
    srv = BankMCPServer("loans", "Lending product information and calculators.", **kwargs)

    @srv.tool(name="get_loan_product_rates", input_model=T.LoanRatesInput, output_model=T.LoanRatesOutput,
              description="Indicative APR range for a lending product. product_type one of: mortgage_30y_fixed, "
                          "mortgage_15y_fixed, mortgage_5_1_arm, auto_new, auto_used, personal, heloc.")
    async def get_loan_product_rates(ctx: ToolContext, product_type: str) -> dict:
        return await gw.loan_rates(product_type)

    @srv.tool(name="calculate_amortization_schedule", input_model=T.AmortizationInput,
              output_model=T.AmortizationOutput,
              description="Compute the fixed monthly payment, total interest and an amortisation preview. "
                          "interest_rate is the ANNUAL percentage (e.g. 6.5), term_months 1-480.")
    async def calculate_amortization_schedule(ctx: ToolContext, principal: float, interest_rate: float,
                                              term_months: int) -> dict:
        return gw.amortization(Decimal(str(principal)), Decimal(str(interest_rate)), term_months)

    @srv.tool(name="fetch_loan_application_status", input_model=T.LoanStatusInput, output_model=T.LoanStatusOutput,
              description="Status and next step of the customer's loan application (APP-######).")
    async def fetch_loan_application_status(ctx: ToolContext, application_id: str) -> dict:
        return await gw.loan_status(ctx.user_id, application_id)

    return srv
