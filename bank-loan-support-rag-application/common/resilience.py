"""Resilience primitives: exponential backoff retries, throttling, circuit breaker.

All three are asyncio-native so they never block the event loop.

* `retry_async`      – exponential backoff with full jitter, honours `Retry-After`.
* `AsyncTokenBucket` – client-side throttling for upstream RPM / TPM quotas.
* `CircuitBreaker`   – CLOSED -> OPEN after N consecutive failures, HALF_OPEN probe
                        after a cool-down; fails fast while OPEN.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TypeVar

import httpx

from common.log import log_event
from common.metrics import RETRIES

T = TypeVar("T")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- errors
class CircuitOpenError(RuntimeError):
    """Raised immediately while a breaker is OPEN."""


class RateLimitedError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, CircuitOpenError):
        return False
    if isinstance(exc, (RateLimitedError, asyncio.TimeoutError, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    # LLM SDKs (groq, google) raise their own classes; inspect generically.
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return any(k in name for k in ("ratelimit", "timeout", "connection", "unavailable", "overloaded"))


def retry_after_seconds(exc: BaseException) -> float | None:
    if isinstance(exc, RateLimitedError):
        return exc.retry_after
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is not None:
        val = headers.get("retry-after") if hasattr(headers, "get") else None
        try:
            return float(val) if val is not None else None
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- retry
async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: Callable[[BaseException], bool] = is_retryable,
    op_name: str = "operation",
) -> T:
    """Run `fn` with exponential backoff + full jitter: sleep ~ U(0, min(max, base*2^n))."""
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except BaseException as exc:  # noqa: BLE001
            last = exc
            if attempt == attempts or not retry_on(exc):
                raise
            delay = random.uniform(0, min(max_delay, base_delay * 2 ** (attempt - 1)))
            ra = retry_after_seconds(exc)
            if ra is not None:
                delay = max(delay, min(ra, max_delay * 4))
            RETRIES.labels(op_name).inc()
            log_event(logger, "retrying", logging.WARNING, op=op_name, attempt=attempt,
                      delay_s=round(delay, 3), error=f"{type(exc).__name__}: {exc}"[:300])
            await asyncio.sleep(delay)
    raise last  # pragma: no cover  (loop always returns or raises)


# --------------------------------------------------------------------------- throttle
class AsyncTokenBucket:
    """Token bucket: `capacity` tokens, refilled continuously at `rate_per_s`.

    `acquire(n)` waits (non-blocking for the loop) until n tokens are available.
    Used twice for the LLM: one bucket for requests/min, one for tokens/min.
    """

    def __init__(self, capacity: float, rate_per_s: float) -> None:
        if capacity <= 0 or rate_per_s <= 0:
            raise ValueError("capacity and rate must be positive")
        self.capacity = float(capacity)
        self.rate = float(rate_per_s)
        self._tokens = float(capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    @classmethod
    def per_minute(cls, n: int) -> AsyncTokenBucket:
        return cls(capacity=n, rate_per_s=n / 60.0)

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    def try_acquire(self, n: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False

    async def acquire(self, n: float = 1.0, timeout: float | None = 30.0) -> None:
        n = min(n, self.capacity)  # a request larger than capacity would never fit
        deadline = None if timeout is None else time.monotonic() + timeout
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return
                wait = (n - self._tokens) / self.rate
                if deadline is not None and time.monotonic() + wait > deadline:
                    raise RateLimitedError("local throttle: quota exhausted", retry_after=wait)
                await asyncio.sleep(min(wait, 1.0))

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens


# --------------------------------------------------------------------------- breaker
class BreakerState(str, Enum):
    closed = "closed"
    open = "open"
    half_open = "half_open"


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, reset_timeout_s: float = 30.0,
                 on_state_change: Callable[[str, BreakerState], None] | None = None) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._state = BreakerState.closed
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_inflight = False
        self._on_change = on_state_change

    @property
    def state(self) -> BreakerState:
        if self._state is BreakerState.open and time.monotonic() - self._opened_at >= self.reset_timeout_s:
            self._set(BreakerState.half_open)
        return self._state

    def _set(self, s: BreakerState) -> None:
        if s is not self._state:
            self._state = s
            log_event(logger, "circuit state change", logging.WARNING, breaker=self.name, state=s.value)
            if self._on_change:
                self._on_change(self.name, s)

    def allow(self) -> bool:
        st = self.state
        if st is BreakerState.closed:
            return True
        if st is BreakerState.half_open and not self._half_open_inflight:
            self._half_open_inflight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._half_open_inflight = False
        self._set(BreakerState.closed)

    def record_failure(self) -> None:
        self._failures += 1
        self._half_open_inflight = False
        if self._state is BreakerState.half_open or self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()
            self._set(BreakerState.open)

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        if not self.allow():
            raise CircuitOpenError(f"circuit '{self.name}' is open")
        try:
            result = await fn()
        except BaseException as exc:
            # Client errors (4xx except 429) say nothing about upstream health.
            if is_retryable(exc) or not isinstance(exc, httpx.HTTPStatusError):
                self.record_failure()
            else:
                self._half_open_inflight = False
            raise
        self.record_success()
        return result


async def resilient_call(
    fn: Callable[[], Awaitable[T]], *, breaker: CircuitBreaker, attempts: int = 4,
    base_delay: float = 0.5, max_delay: float = 8.0, op_name: str = "call",
) -> T:
    """Retry *outside* the breaker so each attempt counts toward tripping it,
    and an OPEN breaker short-circuits remaining attempts."""
    return await retry_async(lambda: breaker.call(fn), attempts=attempts, base_delay=base_delay,
                             max_delay=max_delay, op_name=op_name)
