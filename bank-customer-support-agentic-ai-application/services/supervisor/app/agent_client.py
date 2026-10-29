"""Supervisor -> sub-agent transport (synchronous request/response over HTTPS between Cloud Run
services, wrapped in signed HandoffEnvelopes).

Resilience per sub-agent: circuit breaker, bulkhead, deadline-bounded timeout, one retry for
transport failures (safe: mutating MCP tools are idempotent by key), and a graceful
``failed`` response instead of an exception so one broken agent never sinks the whole plan.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from bankcore.config import Settings, get_settings
from bankcore.errors import BankAgentError, CircuitOpenError, HandoffVerificationError
from bankcore.logging import get_logger
from bankcore.resilience.bulkhead import bulkhead, remaining
from bankcore.resilience.circuit_breaker import get_breaker
from bankcore.resilience.retry import retry_async
from bankcore.schemas.common import AgentName, ErrorInfo
from bankcore.schemas.messages import HandoffEnvelope, SubAgentRequest, SubAgentResponse
from bankcore.security.service_auth import outbound_service_token
from bankcore.security.signing import open_envelope, seal_envelope

log = get_logger(__name__)


def failed_response(req: SubAgentRequest, agent: AgentName, code: str, message: str,
                    retryable: bool = True) -> SubAgentResponse:
    return SubAgentResponse(task_id=req.task_id, agent=agent, status="failed", answer=message,
                            error=ErrorInfo(code=code, message=message, retryable=retryable))


class SubAgentClient(ABC):
    @abstractmethod
    async def invoke(self, agent: AgentName, req: SubAgentRequest, *, trace_id: str) -> SubAgentResponse: ...


class HttpSubAgentClient(SubAgentClient):
    def __init__(self, settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None,
                 urls: dict[str, str] | None = None) -> None:
        self.s = settings or get_settings()
        self.urls = urls or self.s.agent_urls
        self.http = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(self.s.subagent_timeout_s,
                                                                                  connect=3.0),
                                      limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))

    async def invoke(self, agent: AgentName, req: SubAgentRequest, *, trace_id: str) -> SubAgentResponse:
        name = agent.value
        recipient = f"agent-{name}"
        breaker = get_breaker(f"agent:{name}")
        env = seal_envelope(req, sender="supervisor", recipient=recipient, trace_id=trace_id,
                            thread_id=req.thread_id, settings=self.s)
        url = self.urls[name].rstrip("/") + "/v1/invoke"

        async def _post() -> dict[str, Any]:
            timeout = min(self.s.subagent_timeout_s, remaining(req.deadline_ts))
            token = outbound_service_token(recipient, caller="supervisor", settings=self.s, url=url)
            r = await self.http.post(url, content=env.model_dump_json(), timeout=timeout,
                                     headers={"authorization": f"Bearer {token}", "content-type": "application/json",
                                              "x-trace-id": trace_id})
            r.raise_for_status()
            return r.json()

        try:
            async with bulkhead(f"supervisor->{name}", self.s.bulkhead_max_concurrency):
                body = await breaker.call(lambda: retry_async(_post, attempts=2))
            reply = HandoffEnvelope.model_validate(body)
            return open_envelope(reply, expected_recipient="supervisor", model=SubAgentResponse,
                                 allowed_senders={recipient}, settings=self.s)
        except HandoffVerificationError as e:
            log.error("sub-agent reply failed verification", agent=name, error=str(e))
            return failed_response(req, agent, "handoff_rejected", "I couldn't verify that step's result.", False)
        except CircuitOpenError:
            return failed_response(req, agent, "circuit_open",
                                   "That service is temporarily unavailable. Please try again in a few minutes.")
        except BankAgentError as e:
            return failed_response(req, agent, e.code, e.user_message, e.retryable)
        except Exception as e:  # noqa: BLE001
            log.warning("sub-agent call failed", agent=name, error=str(e)[:300])
            return failed_response(req, agent, "agent_unreachable",
                                   "That service is temporarily unavailable. Please try again shortly.")


class InProcessSubAgentClient(SubAgentClient):
    """Runs sub-agents in the same process (tests, local 'monolith' mode). Still seals/opens
    envelopes so handoff verification is exercised end-to-end."""

    def __init__(self, agents: dict[AgentName, Any], settings: Settings | None = None) -> None:
        self.agents, self.s = agents, settings or get_settings()
        self.calls: list[tuple[AgentName, SubAgentRequest]] = []

    async def invoke(self, agent: AgentName, req: SubAgentRequest, *, trace_id: str) -> SubAgentResponse:
        self.calls.append((agent, req))
        recipient = f"agent-{agent.value}"
        env = seal_envelope(req, sender="supervisor", recipient=recipient, trace_id=trace_id,
                            thread_id=req.thread_id, settings=self.s)
        verified = open_envelope(env, expected_recipient=recipient, model=SubAgentRequest,
                                 allowed_senders={"supervisor"}, settings=self.s)
        impl = self.agents.get(agent)
        if impl is None:
            return failed_response(req, agent, "agent_unavailable", "That service is not available.")
        try:
            resp = await impl.run(verified, trace_id=trace_id)
        except Exception as e:  # noqa: BLE001 — same contract as the HTTP client: never raise
            log.warning("in-process sub-agent crashed", agent=agent.value, error=str(e)[:200])
            return failed_response(req, agent, "agent_unreachable",
                                   "That service is temporarily unavailable. Please try again shortly.")
        back = seal_envelope(resp, sender=recipient, recipient="supervisor", trace_id=trace_id,
                             thread_id=req.thread_id, settings=self.s)
        return open_envelope(back, expected_recipient="supervisor", model=SubAgentResponse,
                             allowed_senders={recipient}, settings=self.s)
