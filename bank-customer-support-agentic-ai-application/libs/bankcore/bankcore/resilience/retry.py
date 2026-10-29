"""Retries with exponential back-off + full jitter, ONLY for retryable/idempotent work.

Money-moving tools are never retried blindly: they carry an idempotency key so a retry
after a network blip replays the original result instead of transferring twice.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from bankcore.errors import BankAgentError, CircuitOpenError

T = TypeVar("T")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, CircuitOpenError):
        return False                       # fail fast — retrying an open circuit is pointless
    if isinstance(exc, BankAgentError):
        return exc.retryable
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 502, 503, 504)
    return isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError))


async def retry_async(fn: Callable[[], Awaitable[T]], *, attempts: int = 3, max_wait_s: float = 4.0) -> T:
    async for attempt in AsyncRetrying(stop=stop_after_attempt(attempts),
                                       wait=wait_random_exponential(multiplier=0.25, max=max_wait_s),
                                       retry=retry_if_exception(_is_retryable), reraise=True):
        with attempt:
            return await fn()
    raise RuntimeError("unreachable")  # pragma: no cover
