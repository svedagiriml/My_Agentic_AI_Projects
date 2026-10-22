"""Semantic cache for supervisor ROUTING decisions (Redis).

Routing a query like "what's my checking balance?" through a 70B CoT planner costs ~1-2 s.
Most traffic is a long tail of paraphrases of a few hundred intents, so we cache the *routing
decision* (which sub-agent(s), in which order) keyed by the embedding of an entity-masked query:

    "send $250 from ACC-100001 to ACC-100002"  ->  "send <AMOUNT> from <ACCOUNT> to <ACCOUNT>"

Only routing is cached — never account data or answers — so a hit can't leak one customer's
data to another. Safety rules: single-intent plans only, confidence >= 0.8, no context-dependent
follow-ups ("yes do it"), versioned by prompt version so prompt changes invalidate the cache.

Backends: RediSearch HNSW vector index (redis-stack / Memorystore with vector search) when
available, otherwise a bounded brute-force scan (fine for a few thousand entries).
"""
from __future__ import annotations

import json
import re
import struct
import time
import uuid

from redis.asyncio import Redis

from bankcore.llm.embeddings import Embedder, cosine
from bankcore.logging import get_logger
from bankcore.observability.metrics import ROUTING_CACHE
from bankcore.schemas.messages import SupervisorPlan

log = get_logger(__name__)

_MASKS = [
    (re.compile(r"\bACC-\d{6}\b"), "<ACCOUNT>"),
    (re.compile(r"\bCRD-\d{6}\b"), "<CARD>"),
    (re.compile(r"\bTXN-[A-Z0-9]+\b"), "<TXN>"),
    (re.compile(r"\b(APP|INV|PAY|FND|USR)-[A-Z0-9]+\b"), "<ID>"),
    (re.compile(r"[$₹€£]\s?\d[\d,]*(\.\d+)?|\b\d[\d,]*(\.\d+)?\s?(dollars|usd|inr|rupees)\b", re.I), "<AMOUNT>"),
    (re.compile(r"\b\d{4}-\d{2}(-\d{2})?\b"), "<DATE>"),
    (re.compile(r"\b\d{5}(-\d{4})?\b"), "<ZIP>"),
    (re.compile(r"\b\d+(\.\d+)?\b"), "<NUM>"),
]
_FOLLOWUP = re.compile(r"^\s*(yes|yeah|yep|no|nope|ok|okay|sure|do it|go ahead|that one|the same|"
                       r"the other one|confirm|cancel)\b", re.I)


def mask_entities(text: str) -> str:
    out = text.strip()
    for pattern, token in _MASKS:
        out = pattern.sub(token, out)
    return re.sub(r"\s+", " ", out).lower()


def is_cacheable_query(text: str) -> bool:
    return len(text.split()) >= 3 and not _FOLLOWUP.match(text)


class SemanticRoutingCache:
    PREFIX = "routecache:"
    INDEX = "idx:routecache"

    def __init__(self, redis: Redis, embedder: Embedder, *, threshold: float = 0.92, ttl_s: int = 21600,
                 version: str = "v1", max_entries: int = 5000) -> None:
        self.r, self.embedder = redis, embedder
        self.threshold, self.ttl, self.version, self.max_entries = threshold, ttl_s, version, max_entries
        self._vector_index: bool | None = None

    async def _ensure_index(self) -> bool:
        if self._vector_index is not None:
            return self._vector_index
        try:
            await self.r.execute_command(
                "FT.CREATE", self.INDEX, "ON", "HASH", "PREFIX", "1", self.PREFIX, "SCHEMA",
                "version", "TAG", "embedding", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32",
                "DIM", str(self.embedder.dim), "DISTANCE_METRIC", "COSINE")
            self._vector_index = True
        except Exception as e:  # noqa: BLE001
            self._vector_index = "already exists" in str(e).lower()
            if not self._vector_index:
                log.info("RediSearch unavailable; semantic cache uses brute-force scan")
        return self._vector_index

    @staticmethod
    def _pack(vec: list[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _unpack(raw: bytes) -> list[float]:
        return list(struct.unpack(f"{len(raw) // 4}f", raw))

    async def lookup(self, query: str) -> tuple[SupervisorPlan, float] | None:
        if not is_cacheable_query(query):
            ROUTING_CACHE.labels("skip").inc()
            return None
        try:
            masked = mask_entities(query)
            vec = await self.embedder.embed_one(masked)
            best_key, best_sim = await self._nearest(vec)
            if best_key is None or best_sim < self.threshold:
                ROUTING_CACHE.labels("miss").inc()
                return None
            plan_json = await self.r.hget(best_key, "plan")
            if plan_json is None:
                return None
            plan = SupervisorPlan.model_validate_json(plan_json)
            # Re-bind the cached route to the CURRENT query (never the cached text).
            task = plan.tasks[0].model_copy(update={"instruction": query[:1000]})
            plan = plan.model_copy(update={"tasks": [task], "reasoning": [*plan.reasoning,
                                                                          f"(routing cache hit, sim={best_sim:.3f})"]})
            await self.r.hincrby(best_key, "hits", 1)
            ROUTING_CACHE.labels("hit").inc()
            return plan, best_sim
        except Exception as e:  # noqa: BLE001 — cache must never break routing
            ROUTING_CACHE.labels("error").inc()
            log.warning("semantic cache lookup failed", error=str(e))
            return None

    async def _nearest(self, vec: list[float]) -> tuple[str | None, float]:
        if await self._ensure_index():
            res = await self.r.execute_command(
                "FT.SEARCH", self.INDEX, f"(@version:{{{self.version}}})=>[KNN 1 @embedding $v AS dist]",
                "PARAMS", "2", "v", self._pack(vec), "RETURN", "1", "dist", "DIALECT", "2")
            if res and res[0] >= 1:
                key = res[1].decode() if isinstance(res[1], bytes) else res[1]
                fields = res[2]
                dist = float(fields[1])
                return key, 1.0 - dist
            return None, 0.0
        best_key, best_sim = None, -1.0
        keys = await self.r.zrevrange(f"{self.PREFIX}members:{self.version}", 0, self.max_entries - 1)
        for key in keys:
            raw = await self.r.hget(key, "embedding")
            if raw is None:
                continue
            sim = cosine(vec, self._unpack(raw))
            if sim > best_sim:
                best_key, best_sim = (key.decode() if isinstance(key, bytes) else key), sim
        return best_key, best_sim

    async def store(self, query: str, plan: SupervisorPlan) -> bool:
        if not (is_cacheable_query(query) and len(plan.tasks) == 1 and not plan.needs_clarification
                and plan.confidence >= 0.8):
            return False
        try:
            masked = mask_entities(query)
            vec = await self.embedder.embed_one(masked)
            key = f"{self.PREFIX}{self.version}:{uuid.uuid4().hex}"
            cached = plan.model_copy(update={"reasoning": plan.reasoning[:3]})
            await self.r.hset(key, mapping={"embedding": self._pack(vec), "plan": cached.model_dump_json(),
                                            "masked": masked, "version": self.version, "hits": 0,
                                            "created": time.time()})
            await self.r.expire(key, self.ttl)
            await self.r.zadd(f"{self.PREFIX}members:{self.version}", {key: time.time()})
            ROUTING_CACHE.labels("store").inc()
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("semantic cache store failed", error=str(e))
            return False

    async def stats(self) -> dict:
        n = await self.r.zcard(f"{self.PREFIX}members:{self.version}")
        return {"entries": n, "threshold": self.threshold, "version": self.version,
                "vector_index": bool(self._vector_index)}


def plan_to_json(plan: SupervisorPlan) -> str:
    return json.dumps(plan.model_dump(mode="json"))
