import asyncio

import httpx
import pytest

from common.resilience import (
    AsyncTokenBucket,
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
    RateLimitedError,
    resilient_call,
    retry_async,
)


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://x")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(status, request=req))


async def test_retry_recovers_from_transient_errors():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(503)
        return "ok"

    assert await retry_async(flaky, attempts=4, base_delay=0.001) == "ok"
    assert calls["n"] == 3


async def test_no_retry_on_client_errors():
    calls = {"n": 0}

    async def bad():
        calls["n"] += 1
        raise _http_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        await retry_async(bad, attempts=4, base_delay=0.001)
    assert calls["n"] == 1


async def test_rate_limit_honours_retry_after():
    calls = {"n": 0}

    async def limited():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitedError("429", retry_after=0.05)
        return 1

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await retry_async(limited, attempts=2, base_delay=0.001, max_delay=0.01)
    assert loop.time() - t0 >= 0.04


async def test_circuit_breaker_opens_then_half_opens_then_closes():
    br = CircuitBreaker("t", failure_threshold=2, reset_timeout_s=0.05)

    async def fail():
        raise _http_error(502)

    async def ok():
        return "ok"

    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            await br.call(fail)
    assert br.state is BreakerState.open
    with pytest.raises(CircuitOpenError):
        await br.call(ok)                       # fails fast, upstream not called
    await asyncio.sleep(0.06)
    assert br.state is BreakerState.half_open
    assert await br.call(ok) == "ok"
    assert br.state is BreakerState.closed


async def test_open_breaker_stops_retries():
    br = CircuitBreaker("t2", failure_threshold=1, reset_timeout_s=10)
    calls = {"n": 0}

    async def fail():
        calls["n"] += 1
        raise _http_error(503)

    with pytest.raises((CircuitOpenError, httpx.HTTPStatusError)):
        await resilient_call(fail, breaker=br, attempts=5, base_delay=0.001)
    assert calls["n"] == 1


async def test_token_bucket_throttles():
    b = AsyncTokenBucket(capacity=2, rate_per_s=20)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    for _ in range(4):
        await b.acquire()
    assert loop.time() - t0 >= 0.08          # 2 extra tokens at 20/s


async def test_token_bucket_gives_up_after_timeout():
    b = AsyncTokenBucket(capacity=1, rate_per_s=0.01)
    await b.acquire()
    with pytest.raises(RateLimitedError):
        await b.acquire(timeout=0.01)
