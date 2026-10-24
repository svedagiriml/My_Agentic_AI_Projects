"""Fault-tolerance primitives: circuit breaker, retry w/ jittered backoff, timeouts, bulkheads."""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TypeVar

from loanflow.common.exceptions import CircuitOpenError, LoanFlowError, ToolTimeout
from loanflow.common.logging import get_logger

T = TypeVar("T")
log = get_logger(__name__)


class CBState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Classic three-state breaker.

    CLOSED -> (N consecutive failures) -> OPEN -> (reset_timeout) -> HALF_OPEN
    HALF_OPEN -> success -> CLOSED ; failure -> OPEN
    Only *retryable/infra* failures trip the breaker; business errors (4xx) do not.
    """

    def __init__(self, name: str, failure_threshold: int = 5, reset_timeout_s: float = 30.0, half_open_max: int = 1):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.half_open_max = half_open_max
        self._state = CBState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_inflight = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CBState:
        if self._state == CBState.OPEN and time.monotonic() - self._opened_at >= self.reset_timeout_s:
            return CBState.HALF_OPEN
        return self._state

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            state = self.state
            if state == CBState.OPEN:
                raise CircuitOpenError(f"circuit '{self.name}' is open", details={"breaker": self.name})
            if state == CBState.HALF_OPEN:
                if self._half_open_inflight >= self.half_open_max:
                    raise CircuitOpenError(f"circuit '{self.name}' half-open probe in flight")
                self._state = CBState.HALF_OPEN
                self._half_open_inflight += 1
        try:
            result = await fn()
        except Exception as exc:
            await self._on_failure(exc)
            raise
        else:
            await self._on_success()
            return result

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state != CBState.CLOSED:
                log.info("circuit closed", extra={"extra_fields": {"breaker": self.name}})
            self._state = CBState.CLOSED
            self._failures = 0
            self._half_open_inflight = 0

    async def _on_failure(self, exc: Exception) -> None:
        trips = not isinstance(exc, LoanFlowError) or exc.retryable
        async with self._lock:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
            if not trips:
                return
            self._failures += 1
            if self._state == CBState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = CBState.OPEN
                self._opened_at = time.monotonic()
                log.warning("circuit opened", extra={"extra_fields": {"breaker": self.name, "error": str(exc)}})


_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, failure_threshold: int = 5, reset_timeout_s: float = 30.0) -> CircuitBreaker:
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name, failure_threshold, reset_timeout_s)
    return _breakers[name]


def all_breakers() -> dict[str, str]:
    return {n: b.state.value for n, b in _breakers.items()}


def reset_breakers() -> None:
    _breakers.clear()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, CircuitOpenError):
        return False  # don't hammer an open circuit
    if isinstance(exc, LoanFlowError):
        return exc.retryable
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError))


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.2,
    max_delay_s: float = 5.0,
    name: str = "op",
) -> T:
    """Exponential backoff with full jitter; only retries retryable errors."""
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            last = exc
            if attempt == attempts or not _is_retryable(exc):
                raise
            delay = random.uniform(0, min(max_delay_s, base_delay_s * 2 ** (attempt - 1)))
            log.warning(
                "retrying", extra={"extra_fields": {"op": name, "attempt": attempt, "delay_s": round(delay, 3), "error": repr(exc)}}
            )
            await asyncio.sleep(delay)
    assert last is not None
    raise last


async def with_timeout(coro: Awaitable[T], timeout_s: float, what: str) -> T:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except TimeoutError as exc:
        raise ToolTimeout(f"{what} timed out after {timeout_s}s") from exc


class Bulkhead:
    """Caps concurrency per dependency so one slow upstream can't exhaust the service."""

    def __init__(self, name: str, max_concurrent: int):
        self.name = name
        self._sem = asyncio.Semaphore(max_concurrent)

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._sem:
            return await fn()
