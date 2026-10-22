"""Entry point for every MCP server container.

    MCP_DOMAIN=payments uvicorn mcp_servers.main:app --port 8080

One image, eight Cloud Run services (mcp-account, mcp-payments, ...), each deployed with
--no-allow-unauthenticated and ingress=internal so only its paired sub-agent can reach it.
"""
from __future__ import annotations

import importlib
import os

from bankcore.config import get_settings
from bankcore.resilience.rate_limiter import RateLimiter

DOMAIN_MODULES = {
    "account": "mcp_servers.accounts.server", "payments": "mcp_servers.payments.server",
    "cards": "mcp_servers.cards.server", "fraud": "mcp_servers.fraud.server",
    "loans": "mcp_servers.loans.server", "investments": "mcp_servers.investments.server",
    "knowledge": "mcp_servers.knowledge.server", "compliance": "mcp_servers.compliance.server",
}


def build(domain: str):
    settings = get_settings()
    redis = None
    try:
        from redis.asyncio import Redis
        redis = Redis.from_url(settings.redis_url, socket_timeout=1.0)
    except Exception:  # noqa: BLE001
        pass
    module = importlib.import_module(DOMAIN_MODULES[domain])
    return module.create(rate_limiter=RateLimiter(redis), settings=settings)


app = build(os.getenv("MCP_DOMAIN", "account")).build_app() if os.getenv("MCP_DOMAIN") else None
