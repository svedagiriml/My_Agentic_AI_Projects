"""Async inter-service HTTP client with auth, retries, circuit breaker and streaming.

On Cloud Run the internal services are deployed with `--no-allow-unauthenticated`;
the caller attaches a Google-signed ID token whose audience is the callee URL.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from common.metrics import breaker_listener
from common.resilience import CircuitBreaker, resilient_call

M = TypeVar("M", bound=BaseModel)


class _IdTokenCache:
    def __init__(self) -> None:
        self._tokens: dict[str, tuple[str, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, audience: str) -> str:
        async with self._lock:
            tok = self._tokens.get(audience)
            if tok and tok[1] > time.time() + 60:
                return tok[0]

            def _fetch() -> str:
                import google.auth.transport.requests
                import google.oauth2.id_token

                return google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), audience)

            token = await asyncio.to_thread(_fetch)
            self._tokens[audience] = (token, time.time() + 3300)  # tokens live 1h
            return token


_ID_TOKENS = _IdTokenCache()


class ServiceClient:
    def __init__(self, name: str, base_url: str, *, timeout_s: float = 60.0, use_id_token: bool = False,
                 attempts: int = 3, base_delay: float = 0.3, max_delay: float = 4.0,
                 breaker_threshold: int = 5, breaker_reset_s: float = 30.0) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.use_id_token = use_id_token
        self.attempts, self.base_delay, self.max_delay = attempts, base_delay, max_delay
        self.breaker = CircuitBreaker(f"svc:{name}", breaker_threshold, breaker_reset_s, breaker_listener)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    async def _headers(self, extra: dict[str, str] | None) -> dict[str, str]:
        h = {"Content-Type": "application/json", **(extra or {})}
        if self.use_id_token:
            h["Authorization"] = f"Bearer {await _ID_TOKENS.get(self.base_url)}"
        return h

    async def post_model(self, path: str, body: BaseModel, response_model: type[M],
                         headers: dict[str, str] | None = None) -> M:
        payload = body.model_dump(mode="json")

        async def _do() -> M:
            r = await self._client.post(path, json=payload, headers=await self._headers(headers))
            r.raise_for_status()
            return response_model.model_validate(r.json())

        return await resilient_call(_do, breaker=self.breaker, attempts=self.attempts,
                                    base_delay=self.base_delay, max_delay=self.max_delay,
                                    op_name=f"{self.name}{path}")

    async def stream_sse(self, path: str, body: BaseModel,
                         headers: dict[str, str] | None = None) -> AsyncIterator[tuple[str, str]]:
        """Yield (event, data) tuples. Connection setup is retried; once bytes flow
        we do not retry (the client has already seen partial output)."""
        payload = body.model_dump(mode="json")
        hdrs = await self._headers(headers)

        async def _open() -> httpx.Response:
            req = self._client.build_request("POST", path, json=payload, headers=hdrs)
            resp = await self._client.send(req, stream=True)
            if resp.status_code >= 400:
                await resp.aread()
                resp.raise_for_status()
            return resp

        resp = await resilient_call(_open, breaker=self.breaker, attempts=self.attempts,
                                    base_delay=self.base_delay, max_delay=self.max_delay,
                                    op_name=f"{self.name}{path}:stream")
        try:
            event, data_lines = "message", []
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].removeprefix(" "))
                elif line == "":
                    if data_lines:
                        yield event, "\n".join(data_lines)
                    event, data_lines = "message", []
            if data_lines:
                yield event, "\n".join(data_lines)
        finally:
            await resp.aclose()

    async def get_json(self, path: str) -> Any:
        async def _do() -> Any:
            r = await self._client.get(path, headers=await self._headers(None))
            r.raise_for_status()
            return r.json()

        return await resilient_call(_do, breaker=self.breaker, attempts=2, op_name=f"{self.name}{path}")

    async def aclose(self) -> None:
        await self._client.aclose()
