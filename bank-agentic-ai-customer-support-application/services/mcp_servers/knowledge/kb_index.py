"""Hybrid retrieval over the bank's knowledge base (RAG).

Ingestion: Markdown policy documents -> heading-aware chunks (~600 chars, 1-sentence overlap)
-> injection scan (poisoned documents are quarantined, not indexed) -> embeddings.
Retrieval: dense (cosine on BGE-small / PGVector HNSW) + sparse (BM25) fused with Reciprocal
Rank Fusion, which is robust to score-scale differences between the two retrievers.

Backends: ``InMemoryKnowledgeIndex`` (default, loads data/kb) and ``PgKnowledgeIndex``
(table ``kb_chunks`` with HNSW index + tsvector, populated by scripts/ingest_kb.py).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from bankcore.guardrails.injection import PromptInjectionDetector
from bankcore.llm.embeddings import Embedder, cosine
from bankcore.logging import get_logger

log = get_logger(__name__)
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in on for and or is are be with by at from what how do does i my your can".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Chunk:
    doc_id: str
    title: str
    content: str
    source: str


def chunk_markdown(doc_id: str, text: str, source: str, max_chars: int = 600) -> list[Chunk]:
    title_match = re.search(r"^#\s+(.+)$", text, re.M)
    doc_title = title_match.group(1).strip() if title_match else doc_id
    chunks: list[Chunk] = []
    for section in re.split(r"\n(?=##\s)", text):
        heading = re.search(r"^##\s+(.+)$", section, re.M)
        sec_title = f"{doc_title} — {heading.group(1).strip()}" if heading else doc_title
        body = re.sub(r"^#{1,6}\s+.+$", "", section, flags=re.M).strip()
        sentences = re.split(r"(?<=[.!?])\s+", body)
        buf: list[str] = []
        for s in sentences:
            if sum(len(x) for x in buf) + len(s) > max_chars and buf:
                chunks.append(Chunk(f"{doc_id}#{len(chunks)}", sec_title, " ".join(buf), source))
                buf = buf[-1:]                                  # 1-sentence overlap
            buf.append(s)
        if buf and " ".join(buf).strip():
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", sec_title, " ".join(buf), source))
    return chunks


class InMemoryKnowledgeIndex:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.avgdl = 1.0
        self.quarantined: list[str] = []
        self._detector = PromptInjectionDetector(threshold=0.6)

    async def ingest_dir(self, path: str | Path) -> int:
        for f in sorted(Path(path).glob("*.md")):
            await self.ingest(f.stem, f.read_text(encoding="utf-8"), source=f.name)
        return len(self.chunks)

    async def ingest(self, doc_id: str, text: str, source: str) -> None:
        new = []
        for c in chunk_markdown(doc_id, text, source):
            if self._detector.scan(c.content, untrusted_source=True).is_injection:
                self.quarantined.append(c.doc_id)
                log.warning("kb chunk quarantined (possible poisoning)", doc_id=c.doc_id)
                continue
            new.append(c)
        if not new:
            return
        vecs = await self.embedder.embed([f"{c.title}. {c.content}" for c in new])
        for c, v in zip(new, vecs):
            toks = tokenize(f"{c.title} {c.content}")
            self.chunks.append(c)
            self.vectors.append(v)
            tf = Counter(toks)
            self.tf.append(tf)
            self.df.update(set(toks))
        self.avgdl = sum(sum(t.values()) for t in self.tf) / max(1, len(self.tf))

    def _bm25(self, q_tokens: list[str], i: int, k1: float = 1.4, b: float = 0.75) -> float:
        tf, n = self.tf[i], len(self.chunks)
        dl = sum(tf.values())
        score = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self.avgdl))
        return score

    async def search(self, query: str, top_k: int = 4, rrf_k: int = 60) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        qv = await self.embedder.embed_one(query)
        qt = tokenize(query)
        dense = sorted(range(len(self.chunks)), key=lambda i: cosine(qv, self.vectors[i]), reverse=True)
        sparse = sorted(range(len(self.chunks)), key=lambda i: self._bm25(qt, i), reverse=True)
        fused: dict[int, float] = {}
        for rank, i in enumerate(dense[:20]):
            fused[i] = fused.get(i, 0.0) + 1 / (rrf_k + rank + 1)
        for rank, i in enumerate(sparse[:20]):
            if self._bm25(qt, i) > 0:
                fused[i] = fused.get(i, 0.0) + 1 / (rrf_k + rank + 1)
        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        # normalise to 0..1 for readability (max possible = 2/(rrf_k+1))
        return [(self.chunks[i], round(s / (2 / (rrf_k + 1)), 4)) for i, s in best]


class PgKnowledgeIndex:  # pragma: no cover - requires Postgres
    def __init__(self, pool, embedder: Embedder) -> None:
        self.pool, self.embedder = pool, embedder

    async def search(self, query: str, top_k: int = 4, rrf_k: int = 60) -> list[tuple[Chunk, float]]:
        import numpy as np
        qv = np.array(await self.embedder.embed_one(query), dtype=np.float32)
        sql = """
        WITH dense AS (
            SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS r
              FROM kb_chunks ORDER BY embedding <=> $1 LIMIT 20),
        sparse AS (
            SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
              FROM kb_chunks, plainto_tsquery('english', $2) q WHERE tsv @@ q LIMIT 20)
        SELECT c.chunk_id, c.title, c.content, c.source,
               COALESCE(1.0/($3 + d.r), 0) + COALESCE(1.0/($3 + s.r), 0) AS score
          FROM kb_chunks c
          LEFT JOIN dense d USING (chunk_id) LEFT JOIN sparse s USING (chunk_id)
         WHERE d.r IS NOT NULL OR s.r IS NOT NULL
         ORDER BY score DESC LIMIT $4"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, qv, query, rrf_k, top_k)
        return [(Chunk(r["chunk_id"], r["title"], r["content"], r["source"]),
                 round(float(r["score"]) / (2 / (rrf_k + 1)), 4)) for r in rows]
