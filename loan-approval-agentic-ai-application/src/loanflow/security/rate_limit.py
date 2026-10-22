"""Distributed rate limiting (Redis fixed-window counter + in-process token bucket).

* `check_rate_limit(identity)` – per-principal limit shared by all Cloud Run instances
  (counter lives in Redis, so scaling out does not multiply the allowance).
* `TokenBucket` – local smoothing for outbound calls to free-tier LLM APIs
  (e.g. Groq free tier RPM limits).
"""
from __future__ import annotations

import asyncio
import time

from loanflow.common.config import get_settings
from loanflow.common.exceptions import RateLimitExceeded
from loanflow.memory.kv import get_kv


async def check_rate_limit(identity: str, limit_per_minute: int | None = None, scope: str = "api") -> int:
    limit = limit_per_minute or get_settings().rate_limit_per_minute
    window = int(time.time() // 60)
    count = await get_kv().incr(f"rl:{scope}:{identity}:{window}", ttl_s=70)
    if count > limit:
        raise RateLimitExceeded(f"rate limit of {limit}/min exceeded", details={"retry_after_s": 60 - int(time.time() % 60)})
    return limit - count


class TokenBucket:
    def __init__(self, rate_per_s: float, capacity: int):
        self.rate = rate_per_s
        self.capacity = capacity
        self.tokens = float(capacity)
        self.ts = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
                self.ts = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)
