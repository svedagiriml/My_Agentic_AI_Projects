"""Supervisor -> sub-agent over real HTTP semantics (ASGI transport): service auth + signed envelopes."""
from __future__ import annotations

import httpx
import pytest

from bankcore.schemas.common import AgentName
from bankcore.schemas.messages import SubAgentRequest
from bankcore.security.signing import issue_delegation_token

from subagent.app.main import create_app
from supervisor.app.agent_client import HttpSubAgentClient

pytestmark = pytest.mark.integration


async def test_http_handoff_roundtrip_and_auth(harness, settings):
    app = create_app(agent=harness.agents[AgentName.ACCOUNT], settings=settings)
    client = HttpSubAgentClient(settings, transport=httpx.ASGITransport(app=app),
                                urls={"account": "http://agent-account"})
    req = SubAgentRequest(task_id="t1", thread_id="th", user_id="USR-1001", instruction="balance of ACC-100001",
                          delegation_token=issue_delegation_token(user_id="USR-1001", agent="account", thread_id="th"))
    resp = await client.invoke(AgentName.ACCOUNT, req, trace_id="tr")
    assert resp.status == "completed" and "5240.75" in resp.answer
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://x") as c:
        assert (await c.post("/v1/invoke", json={})).status_code == 401          # no service token


async def test_unreachable_agent_degrades_gracefully(settings):
    client = HttpSubAgentClient(settings, urls={"account": "http://127.0.0.1:1"})
    req = SubAgentRequest(task_id="t1", thread_id="th", user_id="USR-1001", instruction="balance",
                          delegation_token="x")
    resp = await client.invoke(AgentName.ACCOUNT, req, trace_id="tr")
    assert resp.status == "failed" and resp.error.retryable
