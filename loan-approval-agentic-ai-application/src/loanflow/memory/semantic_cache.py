"""Semantic cache (Redis) for supervisor routing decisions.

The supervisor's situation is summarised into a short canonical description
("completed: fetch_document, validate_document | doc_valid=True | credit=pending ...").
Many runs share the same situation, so the Chain-of-Thought LLM call is skipped
when a cached situation is semantically close enough (cosine >= threshold).

Lookup order: exact-hash hit (O(1))  ->  vector similarity over the namespace.
The candidate set is small by design (routing situations are low-cardinality
because the summary contains booleans/buckets, never raw PII or amounts), so a
brute-force scan is cheap; swap in a RediSearch HNSW index if it grows.

Safety: cached decisions are re-validated by the supervisor's transition policy
before use, so a poisoned/stale cache entry can never force an illegal route.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

from loanflow.common.config import get_settings
from loanflow.llm.embeddings import cosine, embed
from loanflow.memory.kv import get_kv


class SemanticCache:
    def __init__(self, namespace: str = "routing", threshold: float | None = None, ttl_s: int | None = None):
        s = get_settings()
        self.ns = namespace
        self.threshold = threshold if threshold is not None else s.semantic_cache_threshold
        self.ttl = ttl_s or s.semantic_cache_ttl_s
        self.kv = get_kv()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _h(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    async def get(self, situation: str) -> tuple[dict[str, Any], float] | None:
        exact = await self.kv.get(f"semcache:{self.ns}:{self._h(situation)}")
        if exact:
            self.hits += 1
            return exact["value"], 1.0
        qv = embed(situation)
        best, best_score = None, 0.0
        for key in await self.kv.keys(f"semcache:{self.ns}:*"):
            entry = await self.kv.get(key)
            if not entry:
                continue
            score = cosine(qv, entry["vec"])
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= self.threshold:
            self.hits += 1
            return best["value"], round(best_score, 4)
        self.misses += 1
        return None

    async def put(self, situation: str, value: dict[str, Any]) -> None:
        await self.kv.set(
            f"semcache:{self.ns}:{self._h(situation)}",
            {"situation": situation, "vec": embed(situation), "value": value, "ts": time.time()},
            ttl_s=self.ttl,
        )

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
