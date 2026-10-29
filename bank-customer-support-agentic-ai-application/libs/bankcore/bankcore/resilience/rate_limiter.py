"""Distributed sliding-window rate limiter backed by Redis (atomic Lua script).

Used at three levels:
* API gateway level   — per customer & per IP (supervisor middleware)
* Tool level          — per customer per sensitive tool (e.g. max 5 transfers / hour)
* LLM provider level  — protect free-tier quotas (Groq RPM/TPM limits)
Falls back to an in-process limiter if Redis is unavailable (fail-open for reads would be
risky, so the fallback is *stricter*: it keeps limiting per instance).
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque

from redis.asyncio import Redis

from bankcore.errors import RateLimitExceeded
from bankcore.logging import get_logger

log = get_logger(__name__)

_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
  return {0, count}
end
redis.call('ZADD', key, now, ARGV[4])
redis.call('PEXPIRE', key, window)
return {1, count + 1}
"""


class RateLimiter:
    def __init__(self, redis: Redis | None) -> None:
        self.redis = redis
        self._local: dict[str, deque[float]] = defaultdict(deque)
        self._script = redis.register_script(_LUA) if redis is not None else None

    async def hit(self, key: str, *, limit: int, window_s: int) -> int:
        """Record a hit; raise RateLimitExceeded when over the limit. Returns current count."""
        now_ms = int(time.time() * 1000)
        if self._script is not None:
            try:
                allowed, count = await self._script(keys=[f"rl:{key}:{window_s}"],
                                                    args=[now_ms, window_s * 1000, limit, uuid.uuid4().hex])
                if not int(allowed):
                    raise RateLimitExceeded(f"rate limit {limit}/{window_s}s exceeded for {key}")
                return int(count)
            except RateLimitExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("rate limiter redis failure, using local fallback", error=str(e))
        q = self._local[f"{key}:{window_s}"]
        while q and q[0] <= now_ms - window_s * 1000:
            q.popleft()
        if len(q) >= limit:
            raise RateLimitExceeded(f"rate limit {limit}/{window_s}s exceeded for {key}")
        q.append(now_ms)
        return len(q)
