"""Semantic cache: reuse answers for questions that *mean* the same thing.

"What is the processing fee for SBI home loan?" and "SBI housing loan processing
charges?" embed to vectors with cosine ≈ 0.95, so the second one is served from
cache in a few ms with zero LLM tokens.

Backends
  * memory – per-instance LRU + TTL (local dev, tests)
  * redis  – Redis Stack / Valkey Search vector index (FLAT exact KNN, cosine),
             shared by all gateway instances. On GCP: Memorystore for Valkey
             (vector search supported) reached through a Serverless VPC connector.

Safety rules (enforced by the orchestrator):
  * never cache when the customer's input contained PII (answer may be personal)
  * never cache blocked, low-confidence or unanswerable responses
  * the key is the *sanitized* standalone question (rewritten query when there is history)
"""
from __future__ import annotations

import json
import struct
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

from common.embeddings import cosine
from common.metrics import CACHE_EVENTS


@dataclass
class CacheHit:
    query: str
    similarity: float
    payload: dict[str, Any]


class SemanticCache(Protocol):
    async def lookup(self, vector: list[float], scope: str = "any") -> CacheHit | None: ...
    async def store(self, query: str, vector: list[float], payload: dict[str, Any], scope: str = "any") -> None: ...


class InMemorySemanticCache:
    def __init__(self, threshold: float, ttl_s: int, max_entries: int) -> None:
        self.threshold, self.ttl_s, self.max_entries = threshold, ttl_s, max_entries
        self._items: OrderedDict[str, tuple[str, list[float], dict[str, Any], float, str]] = OrderedDict()

    async def lookup(self, vector: list[float], scope: str = "any") -> CacheHit | None:
        now = time.time()
        best: tuple[float, str] | None = None
        for key, (_q, vec, _p, ts, sc) in list(self._items.items()):
            if now - ts > self.ttl_s:
                del self._items[key]
                continue
            if sc != scope:
                continue
            sim = cosine(vector, vec)
            if best is None or sim > best[0]:
                best = (sim, key)
        if best and best[0] >= self.threshold:
            self._items.move_to_end(best[1])
            q, _v, p, _ts, _sc = self._items[best[1]]
            CACHE_EVENTS.labels("hit").inc()
            return CacheHit(q, round(best[0], 4), p)
        CACHE_EVENTS.labels("miss").inc()
        return None

    async def store(self, query: str, vector: list[float], payload: dict[str, Any], scope: str = "any") -> None:
        self._items[uuid.uuid4().hex] = (query, vector, payload, time.time(), scope)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
        CACHE_EVENTS.labels("store").inc()


class RedisSemanticCache:
    PREFIX = "semcache:"
    INDEX = "idx:semcache"

    def __init__(self, url: str, dim: int, threshold: float, ttl_s: int) -> None:
        from redis.asyncio import Redis

        self.r = Redis.from_url(url, decode_responses=False)
        self.dim, self.threshold, self.ttl_s = dim, threshold, ttl_s
        self._ready = False

    async def _ensure_index(self) -> None:
        if self._ready:
            return
        from redis.commands.search.field import TagField, TextField, VectorField

        try:
            from redis.commands.search.index_definition import IndexDefinition, IndexType
        except ImportError:  # redis-py < 6
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType  # type: ignore
        try:
            await self.r.ft(self.INDEX).info()
        except Exception:  # noqa: BLE001 - index does not exist yet
            await self.r.ft(self.INDEX).create_index(
                [TagField("scope"), TextField("query"),
                 VectorField("vec", "FLAT", {"TYPE": "FLOAT32", "DIM": self.dim, "DISTANCE_METRIC": "COSINE"})],
                definition=IndexDefinition(prefix=[self.PREFIX], index_type=IndexType.HASH),
            )
        self._ready = True

    def _pack(self, v: list[float]) -> bytes:
        return struct.pack(f"{len(v)}f", *v)

    async def lookup(self, vector: list[float], scope: str = "any") -> CacheHit | None:
        from redis.commands.search.query import Query

        await self._ensure_index()
        q = (Query(f"(@scope:{{{scope}}})=>[KNN 1 @vec $v AS dist]")
             .sort_by("dist").return_fields("query", "payload", "dist").dialect(2))
        res = await self.r.ft(self.INDEX).search(q, query_params={"v": self._pack(vector)})
        if res.docs:
            d = res.docs[0]
            sim = 1.0 - float(d.dist)
            if sim >= self.threshold:
                CACHE_EVENTS.labels("hit").inc()
                return CacheHit(_s(d.query), round(sim, 4), json.loads(_s(d.payload)))
        CACHE_EVENTS.labels("miss").inc()
        return None

    async def store(self, query: str, vector: list[float], payload: dict[str, Any], scope: str = "any") -> None:
        await self._ensure_index()
        key = f"{self.PREFIX}{uuid.uuid4().hex}"
        await self.r.hset(key, mapping={"query": query, "scope": scope, "vec": self._pack(vector),
                                        "payload": json.dumps(payload, default=str)})
        await self.r.expire(key, self.ttl_s)
        CACHE_EVENTS.labels("store").inc()


def _s(x: Any) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)
