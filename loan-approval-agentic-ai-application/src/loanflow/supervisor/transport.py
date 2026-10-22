"""How the supervisor reaches subagents.

* InProcessTransport – same JSON wire format, signing and verification as HTTP, but the
  agent runs in this process (unit/integration tests, `make run-local`).
* HttpTransport      – POST signed AgentTask to the agent's Cloud Run URL with a Google
  ID token (or internal token), per-agent circuit breaker, retry and timeout.
Both verify the reply (schema + checksum + HMAC) before the supervisor trusts it.
"""
from __future__ import annotations

from typing import Protocol

import httpx

from loanflow.common.config import get_settings
from loanflow.common.exceptions import UpstreamUnavailable
from loanflow.common.integrity import parse_and_verify
from loanflow.common.schemas import AgentResult, AgentTask
from loanflow.resilience import get_breaker, retry_async, with_timeout


class AgentTransport(Protocol):
    async def invoke(self, task: AgentTask) -> AgentResult: ...


class InProcessTransport:
    async def invoke(self, task: AgentTask) -> AgentResult:
        from loanflow.agents.service import execute_task

        received = parse_and_verify(AgentTask, task.model_dump_json())  # agent-side verification
        result = await execute_task(received)
        return parse_and_verify(AgentResult, result.model_dump_json())   # supervisor-side verification


class HttpTransport:
    def __init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=get_settings().agent_timeout_s + 5,
                                       limits=httpx.Limits(max_connections=100, max_keepalive_connections=20))

    def _headers(self, url: str) -> dict[str, str]:
        s = get_settings()
        if s.use_gcp_id_tokens:
            from google.auth.transport.requests import Request
            from google.oauth2 import id_token

            return {"Authorization": f"Bearer {id_token.fetch_id_token(Request(), url)}"}
        return {"Authorization": f"Bearer {s.internal_service_token}"}

    async def invoke(self, task: AgentTask) -> AgentResult:
        s = get_settings()
        base = s.agent_urls.get(task.recipient)
        if not base:
            raise UpstreamUnavailable(f"AGENT_URLS has no entry for {task.recipient}")

        async def post() -> AgentResult:
            r = await self._http.post(f"{base.rstrip('/')}/v1/tasks", content=task.model_dump_json(),
                                      headers={**self._headers(base), "content-type": "application/json",
                                               "x-request-id": task.message_id})
            if r.status_code in (429, 502, 503, 504):
                raise UpstreamUnavailable(f"{task.recipient} HTTP {r.status_code}")
            r.raise_for_status()
            return parse_and_verify(AgentResult, r.content)

        breaker = get_breaker(f"agent:{task.recipient}", s.cb_failure_threshold, s.cb_reset_timeout_s)
        return await breaker.call(lambda: retry_async(
            lambda: with_timeout(post(), task.deadline_s + 5, f"agent {task.recipient}"),
            attempts=s.max_retries, name=f"agent:{task.recipient}"))


def get_transport() -> AgentTransport:
    return HttpTransport() if get_settings().transport_mode == "http" else InProcessTransport()
