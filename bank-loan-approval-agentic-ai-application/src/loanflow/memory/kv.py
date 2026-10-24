"""Async key-value abstraction over Redis with an in-memory fallback.

The fallback keeps local dev and the test-suite infra-free; in staging/prod
`REDIS_URL` must be set (Memorystore for Redis) and `validate_for_prod` enforces it.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import time
from typing import Any, Protocol

from loanflow.common.config import get_settings
from loanflow.common.logging import get_logger

log = get_logger(__name__)


class KV(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl_s: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def incr(self, key: str, ttl_s: int) -> int: ...
    async def rpush(self, key: str, value: Any, ttl_s: int | None = None, maxlen: int | None = None) -> None: ...
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> list[Any]: ...
    async def keys(self, pattern: str) -> list[str]: ...
    async def ping(self) -> bool: ...


class InMemoryKV:
    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float | None]] = {}
        self._lock = asyncio.Lock()

    def _alive(self, key: str) -> bool:
        item = self._data.get(key)
        if item is None:
            return False
        if item[1] is not None and item[1] < time.monotonic():
            self._data.pop(key, None)
            return False
        return True

    async def get(self, key: str) -> Any | None:
        return self._data[key][0] if self._alive(key) else None

    async def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        self._data[key] = (json.loads(json.dumps(value, default=str)), time.monotonic() + ttl_s if ttl_s else None)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def incr(self, key: str, ttl_s: int) -> int:
        async with self._lock:
            cur = (await self.get(key)) or 0
            exp = self._data[key][1] if self._alive(key) else time.monotonic() + ttl_s
            self._data[key] = (cur + 1, exp)
            return cur + 1

    async def rpush(self, key: str, value: Any, ttl_s: int | None = None, maxlen: int | None = None) -> None:
        async with self._lock:
            lst = (await self.get(key)) or []
            lst.append(json.loads(json.dumps(value, default=str)))
            if maxlen:
                lst = lst[-maxlen:]
            await self.set(key, lst, ttl_s)

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> list[Any]:
        lst = (await self.get(key)) or []
        return lst[start:] if end == -1 else lst[start : end + 1]

    async def keys(self, pattern: str) -> list[str]:
        return [k for k in list(self._data) if fnmatch.fnmatch(k, pattern) and self._alive(k)]

    async def ping(self) -> bool:
        return True


class RedisKV:
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis  # lazy import

        self._r = redis.from_url(url, decode_responses=True, socket_timeout=2.0, socket_connect_timeout=2.0,
                                 health_check_interval=30, retry_on_timeout=True)

    async def get(self, key: str) -> Any | None:
        raw = await self._r.get(key)
        return json.loads(raw) if raw is not None else None

    async def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        await self._r.set(key, json.dumps(value, default=str), ex=ttl_s)

    async def delete(self, key: str) -> None:
        await self._r.delete(key)

    async def incr(self, key: str, ttl_s: int) -> int:
        async with self._r.pipeline(transaction=True) as p:
            p.incr(key)
            p.expire(key, ttl_s, nx=True)
            n, _ = await p.execute()
        return int(n)

    async def rpush(self, key: str, value: Any, ttl_s: int | None = None, maxlen: int | None = None) -> None:
        async with self._r.pipeline(transaction=True) as p:
            p.rpush(key, json.dumps(value, default=str))
            if maxlen:
                p.ltrim(key, -maxlen, -1)
            if ttl_s:
                p.expire(key, ttl_s)
            await p.execute()

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> list[Any]:
        return [json.loads(x) for x in await self._r.lrange(key, start, end)]

    async def keys(self, pattern: str) -> list[str]:
        return [k async for k in self._r.scan_iter(match=pattern, count=500)]

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except Exception:  # noqa: BLE001
            return False

    @property
    def raw(self):
        return self._r


_kv: KV | None = None


def get_kv() -> KV:
    global _kv
    if _kv is None:
        url = get_settings().redis_url
        if url:
            _kv = RedisKV(url)
        else:
            log.warning("REDIS_URL not set - using in-memory KV (dev/test only)")
            _kv = InMemoryKV()
    return _kv


def set_kv(kv: KV | None) -> None:
    global _kv
    _kv = kv
