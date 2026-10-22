"""MCP server: Fraud & Security Escalation (HIGH RISK — destructive tools need approval)."""
from __future__ import annotations

from bankcore.core_banking.gateway import MockCoreBanking, get_gateway
from bankcore.mcp.server import BankMCPServer, ToolContext
from bankcore.schemas import tools as T


def create(gateway: MockCoreBanking | None = None, **kwargs) -> BankMCPServer:
    gw = gateway or get_gateway()
    srv = BankMCPServer("fraud", "Security containment for the authenticated customer only (user_id USR-####).",
                        tool_rate_limits={"trigger_mfa_challenge": (5, 900)}, **kwargs)

    @srv.tool(name="flag_fraudulent_transaction", input_model=T.FlagFraudInput, output_model=T.FlagFraudOutput,
              high_risk=True, description="Open a fraud investigation case for a disputed transaction (TXN-...).")
    async def flag_fraudulent_transaction(ctx: ToolContext, transaction_id: str, reason: str) -> dict:
        return await gw.flag_fraud(ctx.user_id, transaction_id, reason)

    @srv.tool(name="lock_user_profile", input_model=T.UserInput, output_model=T.LockProfileOutput, high_risk=True,
              description="Lock the customer's online profile and revoke ALL active sessions (account takeover).")
    async def lock_user_profile(ctx: ToolContext, user_id: str) -> dict:
        return await gw.lock_profile(ctx.user_id, user_id)

    @srv.tool(name="trigger_mfa_challenge", input_model=T.MfaInput, output_model=T.MfaOutput,
              description="Send a step-up MFA challenge to the customer via sms, email, push or voice.")
    async def trigger_mfa_challenge(ctx: ToolContext, user_id: str, channel_type: str) -> dict:
        return await gw.mfa(ctx.user_id, user_id, channel_type)

    @srv.tool(name="fetch_active_sessions", input_model=T.UserInput, output_model=T.ActiveSessionsOutput,
              description="List the customer's active login sessions (device, masked IP, location, last seen).")
    async def fetch_active_sessions(ctx: ToolContext, user_id: str) -> dict:
        return await gw.sessions(ctx.user_id, user_id)

    return srv
