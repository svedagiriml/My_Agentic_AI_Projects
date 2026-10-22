"""Circuit breaker, retries, rate limiting, bulkheads, deadlines."""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from bankcore.errors import BankingDomainError, CircuitOpenError, DependencyError, DependencyTimeout, RateLimitExceeded
from bankcore.resilience import CircuitBreaker, RateLimiter, bulkhead, remaining, retry_async, with_timeout
from bankcore.resilience.circuit_breaker import CBState


async def _fail():
    raise ConnectionError("down")


async def _ok():
    return "ok"


async def test_breaker_opens_then_half_open_then_closes():
    cb = CircuitBreaker("t", failure_threshold=2, recovery_timeout_s=0.1)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            await cb.call(_fail)
    assert cb.state == CBState.OPEN
    with pytest.raises(CircuitOpenError):
        await cb.call(_ok)
    await asyncio.sleep(0.12)
    assert cb.state == CBState.HALF_OPEN
    assert await cb.call(_ok) == "ok"
    assert cb.state == CBState.CLOSED


async def test_half_open_failure_reopens():
    cb = CircuitBreaker("t2", failure_threshold=1, recovery_timeout_s=0.05)
    with pytest.raises(ConnectionError):
        await cb.call(_fail)
    await asyncio.sleep(0.06)
    with pytest.raises(ConnectionError):
        await cb.call(_fail)
    assert cb.state == CBState.OPEN


async def test_business_errors_do_not_trip_breaker():
    cb = CircuitBreaker("t3", failure_threshold=1)

    async def insufficient():
        raise BankingDomainError("insufficient funds")
    for _ in range(3):
        with pytest.raises(BankingDomainError):
            await cb.call(insufficient)
    assert cb.state == CBState.CLOSED


async def test_retry_only_retryable():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("blip")
        return "ok"
    assert await retry_async(flaky, attempts=3, max_wait_s=0.01) == "ok"

    calls["n"] = 0

    async def domain():
        calls["n"] += 1
        raise BankingDomainError("no")
    with pytest.raises(BankingDomainError):
        await retry_async(domain, attempts=3, max_wait_s=0.01)
    assert calls["n"] == 1


async def test_local_rate_limiter():
    rl = RateLimiter(None)
    for _ in range(3):
        await rl.hit("u1", limit=3, window_s=60)
    with pytest.raises(RateLimitExceeded):
        await rl.hit("u1", limit=3, window_s=60)
    await rl.hit("u2", limit=3, window_s=60)       # other keys unaffected


async def test_timeout_and_deadline():
    async def slow():
        await asyncio.sleep(1)
    with pytest.raises(DependencyTimeout):
        await with_timeout(slow, 0.05, "slow")
    with pytest.raises(DependencyTimeout):
        remaining(time.time() + 0.1)
    assert remaining(time.time() + 10) > 9


async def test_bulkhead_saturation():
    async def hold(ev):
        async with bulkhead("bh", max_concurrency=1):
            await ev.wait()
    ev = asyncio.Event()
    t = asyncio.create_task(hold(ev))
    await asyncio.sleep(0.01)
    with pytest.raises(DependencyError):
        async with bulkhead("bh", max_concurrency=1, acquire_timeout_s=0.05):
            pass
    ev.set()
    await t
