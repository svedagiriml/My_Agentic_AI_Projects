"""Retriever microservice (FastAPI, async).

POST /retrieve      hybrid search (dense + BM25 + RRF), logs retrieved chunks, LangFuse span
POST /embed         query embeddings (used by the gateway's semantic cache, so the
                    embedding model is loaded in exactly one service)
GET  /admin/reload  rebuild the BM25 index after an ingestion run
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException

from common import storage
from common.app_factory import create_app
from common.config import get_settings
from common.embeddings import build_embedder
from common.log import log_event
from common.observability import PARENT_HEADER, TRACE_HEADER, Tracer
from common.schemas import EmbedRequest, EmbedResponse, RetrieveRequest, RetrieveResponse
from common.vectorstore import build_vector_store
from services.retriever.retrieval_logger import RetrievalLogger
from services.retriever.service import HybridRetriever

settings = get_settings()
state: dict[str, object] = {}
logger = logging.getLogger("retriever")


async def _watch_corpus(retriever: HybridRetriever, every_s: float = 60.0) -> None:
    """Every instance polls the version marker written by ingestion, so a CDC run
    reaches all retriever replicas (not just the one that got /admin/reload)."""
    marker = settings.chunk_store_uri + ".version"
    seen = await asyncio.to_thread(storage.read_text, marker)
    while True:
        await asyncio.sleep(every_s)
        try:
            cur = await asyncio.to_thread(storage.read_text, marker)
            if cur and cur != seen:
                await asyncio.to_thread(retriever.reload)
                seen = cur
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "corpus watch failed", logging.WARNING, error=str(exc))


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    embedder = await asyncio.to_thread(build_embedder, settings)
    store = await asyncio.to_thread(build_vector_store, settings)
    retriever = HybridRetriever(settings, store, embedder)
    await asyncio.to_thread(retriever.reload)
    state.update(retriever=retriever, embedder=embedder, tracer=Tracer(settings, "retriever"),
                 rlog=RetrievalLogger(settings.retrieval_log_path))
    watcher = asyncio.create_task(_watch_corpus(retriever))
    yield
    watcher.cancel()
    state["tracer"].flush()  # type: ignore[attr-defined]


app = create_app("retriever", settings, lifespan)


def _r() -> HybridRetriever:
    r = state.get("retriever")
    if r is None:
        raise HTTPException(503, "retriever warming up")
    return r  # type: ignore[return-value]


@app.get("/readyz")
async def ready() -> dict[str, object]:
    r = _r()
    return {"ready": True, "bm25_chunks": len(r.bm25), "pinecone_breaker": r.pinecone_breaker.state.value}


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest, x_trace_id: str | None = Header(None, alias=TRACE_HEADER),
                   x_parent: str | None = Header(None, alias=PARENT_HEADER)) -> RetrieveResponse:
    tracer: Tracer = state["tracer"]  # type: ignore[assignment]
    trace_id = req.trace_id or x_trace_id
    with tracer.span("retriever.hybrid_search", trace_id=trace_id, parent_span_id=x_parent,
                     input={"query": req.query, "product_filter": req.product_filter, "top_k": req.top_k}) as span:
        resp = await _r().retrieve(req)
        span.update(output={
            "retrieved": [{"chunk_id": c.chunk_id, "rank": c.final_rank, "rrf_score": c.rrf_score,
                           "dense_score": c.dense_score, "dense_rank": c.dense_rank,
                           "bm25_score": c.bm25_score, "bm25_rank": c.bm25_rank,
                           "source": c.source_file, "page": c.page} for c in resp.chunks],
        }, metadata={"latency_ms": resp.latency_ms, "dense_candidates": resp.dense_candidates,
                     "bm25_candidates": resp.bm25_candidates, "rrf_k": settings.rrf_k})
    await state["rlog"].log(resp, trace_id=trace_id,  # type: ignore[attr-defined]
                            product_filter=req.product_filter.value if req.product_filter else None)
    return resp


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    emb = state["embedder"]
    if req.is_query:
        vecs = await asyncio.gather(*(asyncio.to_thread(emb.embed_query, t) for t in req.texts))  # type: ignore[attr-defined]
    else:
        vecs = await asyncio.to_thread(emb.embed_documents, req.texts)  # type: ignore[attr-defined]
    return EmbedResponse(vectors=list(vecs), dim=len(vecs[0]))


@app.get("/admin/reload")
async def reload() -> dict[str, int]:
    n = await asyncio.to_thread(_r().reload)
    return {"chunks": n}
