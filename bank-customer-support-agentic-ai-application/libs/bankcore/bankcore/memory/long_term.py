"""Long-term memory on Postgres + PGVector (episodic, semantic, procedural).

Tables (see db/init.sql): ``episodic_memory``, ``semantic_memory``, ``procedural_memory``,
``memory_quarantine``. Every write passes ``MemoryGuard``; every read is hard-filtered by
user namespace and expiry, then re-ranked by similarity x trust x recency.
"""
from __future__ import annotations

import json
import math
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from bankcore.guardrails.memory_guard import MemoryGuard
from bankcore.llm.embeddings import Embedder, cosine
from bankcore.logging import get_logger
from bankcore.observability.metrics import MEMORY_OPS
from bankcore.schemas.memory import MemoryKind, MemoryRecord, RetrievedMemory

log = get_logger(__name__)

TABLES = {MemoryKind.EPISODIC: "episodic_memory", MemoryKind.SEMANTIC: "semantic_memory",
          MemoryKind.PROCEDURAL: "procedural_memory"}
_HALF_LIFE_DAYS = {MemoryKind.EPISODIC: 30.0, MemoryKind.SEMANTIC: 365.0, MemoryKind.PROCEDURAL: 180.0}


def rerank_score(similarity: float, trust: float, created_at: datetime, kind: MemoryKind) -> float:
    age_days = max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds() / 86400)
    recency = math.pow(0.5, age_days / _HALF_LIFE_DAYS[kind])
    return similarity * (0.6 + 0.25 * trust + 0.15 * recency)


class LongTermMemory(ABC):
    def __init__(self, embedder: Embedder, guard: MemoryGuard | None = None) -> None:
        self.embedder = embedder
        self.guard = guard or MemoryGuard()

    async def write(self, record: MemoryRecord) -> str | None:
        decision = self.guard.check(record)
        if not decision.allowed or decision.record is None:
            MEMORY_OPS.labels("pgvector", f"write_{record.kind.value}", "quarantined").inc()
            log.warning("memory write quarantined", kind=record.kind.value, reason=decision.reason)
            await self._quarantine(record, decision.reason)
            return None
        rec = decision.record
        emb = await self.embedder.embed_one(rec.content)
        mid = await self._insert(rec, emb)
        MEMORY_OPS.labels("pgvector", f"write_{rec.kind.value}", "ok").inc()
        return mid

    async def search(self, *, user_id: str, query: str, kinds: list[MemoryKind], k: int = 5,
                     min_trust: float = 0.5, min_similarity: float = 0.25) -> list[RetrievedMemory]:
        emb = await self.embedder.embed_one(query)
        rows = await self._query(user_id, emb, kinds, k * 3, min_trust)
        ranked = [r for r in rows if r.score >= min_similarity]
        ranked.sort(key=lambda r: rerank_score(r.score, r.trust_score, r.created_at, r.kind), reverse=True)
        MEMORY_OPS.labels("pgvector", "search", "ok").inc()
        return ranked[:k]

    @abstractmethod
    async def _insert(self, rec: MemoryRecord, embedding: list[float]) -> str: ...

    @abstractmethod
    async def _query(self, user_id: str, embedding: list[float], kinds: list[MemoryKind], k: int,
                     min_trust: float) -> list[RetrievedMemory]: ...

    @abstractmethod
    async def _quarantine(self, rec: MemoryRecord, reason: str) -> None: ...

    @abstractmethod
    async def forget_user(self, user_id: str) -> int:
        """GDPR / CCPA / DPDP right-to-erasure."""


class PgVectorLongTermMemory(LongTermMemory):
    def __init__(self, pool, embedder: Embedder, guard: MemoryGuard | None = None) -> None:
        super().__init__(embedder, guard)
        self.pool = pool

    @classmethod
    async def create(cls, dsn: str, embedder: Embedder, guard: MemoryGuard | None = None):
        import asyncpg
        from pgvector.asyncpg import register_vector

        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10, command_timeout=5, init=register_vector)
        return cls(pool, embedder, guard)

    async def _insert(self, rec: MemoryRecord, embedding: list[float]) -> str:
        import numpy as np

        table = TABLES[rec.kind]
        mid = uuid.uuid4().hex
        sql = f"""
            INSERT INTO {table} (id, user_id, content, metadata, source, trust_score, trace_id,
                                 content_sha256, embedding, created_at, expires_at)
            VALUES ($1,$2,$3,$4::jsonb,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (content_sha256) DO UPDATE
               SET hit_count = {table}.hit_count + 1, last_seen_at = now(),
                   trust_score = GREATEST({table}.trust_score, EXCLUDED.trust_score)
            RETURNING id"""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(sql, mid, rec.user_id, rec.content, json.dumps(rec.metadata),
                                       rec.source.value, rec.trust_score, rec.trace_id, rec.content_sha256,
                                       np.array(embedding, dtype=np.float32), rec.created_at, rec.expires_at)

    async def _query(self, user_id: str, embedding: list[float], kinds: list[MemoryKind], k: int,
                     min_trust: float) -> list[RetrievedMemory]:
        import numpy as np

        vec = np.array(embedding, dtype=np.float32)
        out: list[RetrievedMemory] = []
        async with self.pool.acquire() as conn:
            for kind in kinds:
                table = TABLES[kind]
                if kind == MemoryKind.PROCEDURAL:          # global, customer-agnostic
                    where, args = "user_id IS NULL AND trust_score >= $2", (vec, min_trust)
                else:                                       # hard tenant isolation
                    where, args = "user_id = $2 AND trust_score >= $3", (vec, user_id, min_trust)
                sql = f"""
                    SELECT content, trust_score, created_at, 1 - (embedding <=> $1) AS sim
                      FROM {table}
                     WHERE {where} AND (expires_at IS NULL OR expires_at > now())
                     ORDER BY embedding <=> $1 LIMIT {int(k)}"""
                for r in await conn.fetch(sql, *args):
                    out.append(RetrievedMemory(kind=kind, content=r["content"], score=float(r["sim"]),
                                               trust_score=float(r["trust_score"]), created_at=r["created_at"]))
        return out

    async def _quarantine(self, rec: MemoryRecord, reason: str) -> None:
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO memory_quarantine (id, user_id, kind, content, source, reason, trace_id) "
                    "VALUES ($1,$2,$3,$4,$5,$6,$7)", uuid.uuid4().hex, rec.user_id, rec.kind.value,
                    rec.content[:2000], rec.source.value, reason, rec.trace_id)
        except Exception as e:  # noqa: BLE001
            log.error("quarantine insert failed", error=str(e))

    async def forget_user(self, user_id: str) -> int:
        total = 0
        async with self.pool.acquire() as conn, conn.transaction():
            for table in ("episodic_memory", "semantic_memory"):
                res = await conn.execute(f"DELETE FROM {table} WHERE user_id = $1", user_id)
                total += int(res.split()[-1])
        return total


class InMemoryLongTermMemory(LongTermMemory):
    """Same semantics as PgVector store; used in tests and as a degraded-mode fallback."""

    def __init__(self, embedder: Embedder, guard: MemoryGuard | None = None) -> None:
        super().__init__(embedder, guard)
        self.rows: dict[str, tuple[MemoryRecord, list[float]]] = {}
        self.quarantine: list[tuple[MemoryRecord, str]] = []

    async def _insert(self, rec: MemoryRecord, embedding: list[float]) -> str:
        for mid, (existing, _) in self.rows.items():
            if existing.content_sha256 == rec.content_sha256:
                return mid
        mid = uuid.uuid4().hex
        self.rows[mid] = (rec.model_copy(update={"id": mid}), embedding)
        return mid

    async def _query(self, user_id: str, embedding: list[float], kinds: list[MemoryKind], k: int,
                     min_trust: float) -> list[RetrievedMemory]:
        now = datetime.now(timezone.utc)
        out = []
        for rec, emb in self.rows.values():
            if rec.kind not in kinds or rec.trust_score < min_trust:
                continue
            if rec.kind == MemoryKind.PROCEDURAL:
                if rec.user_id is not None:
                    continue
            elif rec.user_id != user_id:
                continue
            if rec.expires_at and rec.expires_at <= now:
                continue
            out.append(RetrievedMemory(kind=rec.kind, content=rec.content, score=cosine(embedding, emb),
                                       trust_score=rec.trust_score, created_at=rec.created_at))
        out.sort(key=lambda r: r.score, reverse=True)
        return out[:k]

    async def _quarantine(self, rec: MemoryRecord, reason: str) -> None:
        self.quarantine.append((rec, reason))

    async def forget_user(self, user_id: str) -> int:
        doomed = [m for m, (r, _) in self.rows.items() if r.user_id == user_id]
        for m in doomed:
            del self.rows[m]
        return len(doomed)
