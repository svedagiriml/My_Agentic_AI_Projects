"""Async circuit breaker (closed -> open -> half-open) with a process-wide registry.

One breaker per downstream dependency (each sub-agent, each MCP server, each LLM provider).
When open, calls fail fast with ``CircuitOpenError`` so the supervisor can degrade gracefully
(fallback provider, keyword router, cached answer, polite apology) instead of piling up
requests on a sick dependency. State is exported as a Prometheus gauge.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TypeVar

from bankcore.errors import BankAgentError, CircuitOpenError
from bankcore.observability.metrics import CIRCUIT_STATE

T = TypeVar("T")


class CBState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


_STATE_VALUE = {CBState.CLOSED: 0, CBState.HALF_OPEN: 1, CBState.OPEN: 2}


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout_s: float = 30.0,
                 half_open_max_calls: int = 1,
                 counts_as_failure: Callable[[BaseException], bool] | None = None) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_calls = half_open_max_calls
        self._counts = counts_as_failure or self._default_is_failure
        self._state = CBState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_inflight = 0
        self._lock = asyncio.Lock()
        CIRCUIT_STATE.labels(name).set(0)

    @staticmethod
    def _default_is_failure(exc: BaseException) -> bool:
        # Business-rule errors (insufficient funds, validation) must NOT trip the breaker.
        if isinstance(exc, BankAgentError):
            return exc.retryable
        # Cancellation is not a dependency failure; everything else unexpected is.
        return not isinstance(exc, asyncio.CancelledError)

    @property
    def state(self) -> CBState:
        if self._state == CBState.OPEN and time.monotonic() - self._opened_at >= self.recovery_timeout_s:
            return CBState.HALF_OPEN
        return self._state

    def _set(self, state: CBState) -> None:
        self._state = state
        CIRCUIT_STATE.labels(self.name).set(_STATE_VALUE[state])

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            state = self.state
            if state == CBState.OPEN:
                raise CircuitOpenError(f"circuit '{self.name}' is open")
            if state == CBState.HALF_OPEN:
                if self._half_open_inflight >= self.half_open_max_calls:
                    raise CircuitOpenError(f"circuit '{self.name}' is half-open (probe in flight)")
                self._set(CBState.HALF_OPEN)
                self._half_open_inflight += 1
        try:
            result = await fn()
        except BaseException as exc:
            async with self._lock:
                if self._state == CBState.HALF_OPEN:
                    self._half_open_inflight = max(0, self._half_open_inflight - 1)
                if self._counts(exc):
                    self._failures += 1
                    if self._state == CBState.HALF_OPEN or self._failures >= self.failure_threshold:
                        self._set(CBState.OPEN)
                        self._opened_at = time.monotonic()
            raise
        async with self._lock:
            if self._state == CBState.HALF_OPEN:
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
            self._failures = 0
            self._set(CBState.CLOSED)
        return result


_registry: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, **kwargs) -> CircuitBreaker:
    if name not in _registry:
        from bankcore.config import get_settings
        s = get_settings()
        kwargs.setdefault("failure_threshold", s.cb_failure_threshold)
        kwargs.setdefault("recovery_timeout_s", s.cb_recovery_s)
        _registry[name] = CircuitBreaker(name, **kwargs)
    return _registry[name]


def reset_breakers() -> None:
    _registry.clear()
