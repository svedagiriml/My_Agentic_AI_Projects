"""Hybrid retriever core (framework-free so it is unit-testable)."""
from __future__ import annotations

import asyncio
import logging
import time

from common.chunk_store import load_chunks
from common.config import Settings
from common.embeddings import Embedder, aembed_query
from common.log import log_event
from common.metrics import RETRIEVAL_LATENCY, RETRIEVAL_TOP_SCORE, breaker_listener
from common.resilience import CircuitBreaker, CircuitOpenError, resilient_call
from common.schemas import Chunk, ChunkType, LoanProduct, RetrievedChunk, RetrieveRequest, RetrieveResponse
from common.vectorstore import VectorHit, VectorStore, aquery
from services.retriever.bm25_index import BM25Index
from services.retriever.hybrid import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, settings: Settings, store: VectorStore, embedder: Embedder) -> None:
        self.s = settings
        self.store = store
        self.embedder = embedder
        self.bm25 = BM25Index()
        self.pinecone_breaker = CircuitBreaker("pinecone", settings.breaker_failure_threshold,
                                               settings.breaker_reset_timeout_s, breaker_listener)

    def reload(self, chunks: list[Chunk] | None = None) -> int:
        chunks = chunks if chunks is not None else load_chunks(self.s.chunk_store_uri)
        self.bm25.build(chunks)
        log_event(logger, "bm25 corpus loaded", chunks=len(chunks))
        return len(chunks)

    @staticmethod
    def _products(p: LoanProduct | None) -> set[str] | None:
        # "general" docs (e.g. RBI KFS circular) apply to every product.
        return None if p in (None, LoanProduct.general) else {p.value, LoanProduct.general.value}

    async def _dense(self, query: str, products: set[str] | None) -> list[VectorHit]:
        t0 = time.perf_counter()
        vec = await aembed_query(self.embedder, query)
        RETRIEVAL_LATENCY.labels("embed").observe(time.perf_counter() - t0)
        flt = {"product": {"$in": sorted(products)}} if products else None
        t1 = time.perf_counter()
        try:
            hits = await resilient_call(lambda: aquery(self.store, vec, self.s.dense_top_k, flt),
                                        breaker=self.pinecone_breaker, attempts=self.s.retry_max_attempts,
                                        base_delay=self.s.retry_base_delay_s, op_name="pinecone.query")
        except Exception as exc:  # noqa: BLE001 - degrade to keyword-only search
            log_event(logger, "dense search unavailable, falling back to BM25 only", logging.ERROR,
                      error=f"{type(exc).__name__}: {exc}", breaker_open=isinstance(exc, CircuitOpenError))
            hits = []
        RETRIEVAL_LATENCY.labels("dense").observe(time.perf_counter() - t1)
        return hits

    async def retrieve(self, req: RetrieveRequest) -> RetrieveResponse:
        t0 = time.perf_counter()
        products = self._products(req.product_filter)

        async def _bm25() -> list[tuple[str, float]]:
            t = time.perf_counter()
            res = await asyncio.to_thread(self.bm25.search, req.query, self.s.bm25_top_k, products)
            RETRIEVAL_LATENCY.labels("bm25").observe(time.perf_counter() - t)
            return res

        dense_hits, bm25_hits = await asyncio.gather(self._dense(req.query, products), _bm25())

        t_rrf = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {"dense": [(h.id, h.score) for h in dense_hits], "bm25": bm25_hits},
            k=self.s.rrf_k, top_n=req.top_k,
        )
        RETRIEVAL_LATENCY.labels("rrf").observe(time.perf_counter() - t_rrf)

        meta = {h.id: h.metadata for h in dense_hits}
        corpus = self.bm25.by_id
        out: list[RetrievedChunk] = []
        for rank, f in enumerate(fused, start=1):
            c = corpus.get(f.chunk_id)
            md = meta.get(f.chunk_id, {})
            if c is None and not md:
                continue
            out.append(RetrievedChunk(
                chunk_id=f.chunk_id,
                doc_id=c.doc_id if c else str(md.get("doc_id", "")),
                source_file=c.source_file if c else str(md.get("source_file", "")),
                chunk_type=c.chunk_type if c else ChunkType(md.get("chunk_type", "paragraph")),
                section_path=c.section_path if c else list(md.get("section_path", [])),
                page=c.page if c else (int(md["page"]) if md.get("page") else None),
                display_text=c.display_text if c else str(md.get("display_text", "")),
                dense_score=round(f.scores["dense"], 6) if "dense" in f.scores else None,
                dense_rank=f.ranks.get("dense"),
                bm25_score=round(f.scores["bm25"], 6) if "bm25" in f.scores else None,
                bm25_rank=f.ranks.get("bm25"),
                rrf_score=round(f.rrf_score, 6),
                final_rank=rank,
            ))
        if out:
            RETRIEVAL_TOP_SCORE.observe(out[0].rrf_score)
        latency = (time.perf_counter() - t0) * 1000
        RETRIEVAL_LATENCY.labels("total").observe(latency / 1000)
        return RetrieveResponse(query=req.query, chunks=out, latency_ms=round(latency, 2),
                                dense_candidates=len(dense_hits), bm25_candidates=len(bm25_hits))
