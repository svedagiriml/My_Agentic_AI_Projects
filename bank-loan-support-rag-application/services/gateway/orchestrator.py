"""Gateway orchestration of one customer query across the microservices.

guardrail(in) -> semantic cache -> rewrite -> hybrid retrieve -> generate(stream) -> guardrail(out) -> cache
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from common.config import Settings
from common.http_client import ServiceClient
from common.log import log_event, trace_id_var
from common.observability import Tracer, new_trace_id, trace_headers
from common.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    GuardrailRequest,
    GuardrailResponse,
    LoanProduct,
    RetrieveRequest,
    RetrieveResponse,
    RewriteRequest,
    RewrittenQuery,
    SSEEvent,
)
from services.gateway.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)

OFF_TOPIC = ("I'm the bank's loan assistant, so I can help with home loans, personal loans, car loans, "
             "their rates, fees, eligibility, documents and repayment. What would you like to know?")
NO_CONTEXT = ("I couldn't find this in our loan documents. Please contact your nearest branch or customer "
              "care so a loan officer can help you.")
UNAVAILABLE = "Our assistant is temporarily busy. Please try again in a minute."
SENTENCE_END = re.compile(r"([.!?:]\s|\n)")


class Orchestrator:
    def __init__(self, settings: Settings, tracer: Tracer, cache: SemanticCache, retriever: ServiceClient,
                 generator: ServiceClient, guardrail: ServiceClient) -> None:
        self.s = settings
        self.tracer = tracer
        self.cache = cache
        self.retriever = retriever
        self.generator = generator
        self.guardrail = guardrail

    # ------------------------------------------------------------------ steps
    async def _guard(self, text: str, direction: str, hdr: dict[str, str]) -> GuardrailResponse:
        return await self.guardrail.post_model("/check", GuardrailRequest(text=text, direction=direction),  # type: ignore[arg-type]
                                               GuardrailResponse, hdr)

    async def _embed(self, text: str, hdr: dict[str, str]) -> list[float]:
        r = await self.retriever.post_model("/embed", EmbedRequest(texts=[text]), EmbedResponse, hdr)
        return r.vectors[0]

    async def _prepare(self, req: ChatRequest, trace_id: str, root: Any) -> dict[str, Any]:
        """Everything before generation. Returns a dict with either `final` (short-circuit
        answer) or the material needed to generate."""
        hdr = trace_headers(trace_id, root)
        g_in = await self._guard(req.query, "input", hdr)
        if g_in.blocked:
            return {"final": g_in.block_reason, "blocked": True, "reason": "guardrail_input"}
        query = g_in.sanitized_text
        has_pii = bool(g_in.pii_found)
        root.update(metadata={"pii_types_in_query": sorted({e.entity_type for e in g_in.pii_found})})

        cache_key, vec = None, None
        cacheable = not has_pii
        if cacheable and not req.history:
            cache_key = query
            with self.tracer.span("semantic_cache.lookup", trace_id=trace_id, parent_span_id=root.id,
                                  input={"query": query}) as sp:
                try:
                    vec = await self._embed(query, hdr)
                    hit = await self.cache.lookup(vec)
                except Exception as exc:  # noqa: BLE001 - cache must never fail the request
                    log_event(logger, "cache lookup failed", logging.WARNING, error=str(exc))
                    hit = None
                sp.update(output={"hit": bool(hit), "similarity": hit.similarity if hit else None,
                                  "matched_query": hit.query if hit else None})
            if hit:
                return {"final": hit.payload["answer"], "citations": hit.payload.get("citations", []),
                        "cached": True, "query": query}

        rw = await self.generator.post_model("/rewrite", RewriteRequest(query=query, history=req.history,
                                                                       trace_id=trace_id),
                                             RewrittenQuery, hdr)
        if not rw.is_banking_question:
            return {"final": OFF_TOPIC, "reason": "off_topic", "query": query, "rewritten": rw}

        product = None if rw.product is LoanProduct.general else rw.product
        rr = await self.retriever.post_model("/retrieve", RetrieveRequest(query=rw.rewritten_query,
                                                                         product_filter=product,
                                                                         top_k=self.s.final_top_k,
                                                                         trace_id=trace_id),
                                             RetrieveResponse, hdr)
        if not rr.chunks:
            return {"final": NO_CONTEXT, "reason": "no_context", "query": query, "rewritten": rw}

        if cacheable and cache_key is None:   # conversational turn: key on the standalone question
            cache_key = rw.rewritten_query
            vec = None
        return {"query": query, "rewritten": rw, "retrieved": rr, "cache_key": cache_key, "vec": vec,
                "cacheable": cacheable, "hdr": hdr}

    async def _maybe_cache(self, prep: dict[str, Any], answer: str, citations: list[Citation],
                           ok: bool) -> None:
        if not (ok and prep.get("cacheable") and prep.get("cache_key") and citations):
            return
        try:
            vec = prep.get("vec") or await self._embed(prep["cache_key"], prep["hdr"])
            await self.cache.store(prep["cache_key"], vec, {"answer": answer,
                                                            "citations": [c.model_dump() for c in citations]})
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "cache store failed", logging.WARNING, error=str(exc))

    def _root_update(self, root: Any, req: ChatRequest, prep: dict[str, Any], answer: str, t0: float) -> None:
        rw: RewrittenQuery | None = prep.get("rewritten")
        rr: RetrieveResponse | None = prep.get("retrieved")
        root.update(output={"answer": answer}, metadata={
            "raw_query_sanitized": prep.get("query"),
            "rewritten_query": rw.rewritten_query if rw else None,
            "product": rw.product.value if rw else None,
            "intent": rw.intent if rw else None,
            "retrieved_chunk_ids": [c.chunk_id for c in rr.chunks] if rr else [],
            "retrieved_scores": [{"id": c.chunk_id, "rrf": c.rrf_score, "dense": c.dense_score,
                                  "bm25": c.bm25_score} for c in rr.chunks] if rr else [],
            "cached": prep.get("cached", False), "short_circuit": prep.get("reason"),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        })
        root.update_trace(name="bank-loan-chat", session_id=req.session_id,
                          tags=["cached"] if prep.get("cached") else ["rag"])

    # ------------------------------------------------------------------ non-streaming
    async def chat(self, req: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()
        trace_id = new_trace_id()
        trace_id_var.set(trace_id)
        with self.tracer.span("chat_request", trace_id=trace_id,
                              input={"query_length": len(req.query), "history_turns": len(req.history)}) as root:
            try:
                prep = await self._prepare(req, trace_id, root)
                if "final" in prep:
                    answer, cits = prep["final"], [Citation(**c) for c in prep.get("citations", [])]
                    conf = "high" if prep.get("cached") else "low"
                else:
                    g = await self.generator.post_model("/generate", GenerateRequest(
                        query=prep["query"], rewritten_query=prep["rewritten"].rewritten_query,
                        chunks=prep["retrieved"].chunks, history=req.history, trace_id=trace_id),
                        GenerateResponse, prep["hdr"])
                    out = await self._guard(g.answer, "output", prep["hdr"])
                    answer, cits, conf = out.sanitized_text, g.citations, g.confidence
                    await self._maybe_cache(prep, answer, cits, ok=not out.blocked and g.answerable_from_context
                                            and g.confidence != "low")
            except Exception as exc:  # noqa: BLE001
                log_event(logger, "chat failed", logging.ERROR, error=f"{type(exc).__name__}: {exc}")
                prep, answer, cits, conf = {"reason": "error"}, UNAVAILABLE, [], "low"
            self._root_update(root, req, prep, answer, t0)
        return ChatResponse(answer=answer, citations=cits, confidence=conf, cached=bool(prep.get("cached")),
                            trace_id=trace_id, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                            blocked=bool(prep.get("blocked")))

    # ------------------------------------------------------------------ streaming
    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[str]:
        t0 = time.perf_counter()
        trace_id = new_trace_id()
        trace_id_var.set(trace_id)
        with self.tracer.span("chat_request", trace_id=trace_id,
                              input={"query_length": len(req.query), "history_turns": len(req.history)}) as root:
            yield SSEEvent(event="meta", data={"trace_id": trace_id}).encode()
            prep: dict[str, Any] = {}
            answer = ""
            try:
                prep = await self._prepare(req, trace_id, root)
                if "final" in prep:
                    answer = prep["final"]
                    yield SSEEvent(event="token", data={"t": answer}).encode()
                    yield SSEEvent(event="citations", data={"citations": prep.get("citations", [])}).encode()
                    yield SSEEvent(event="done", data={"cached": bool(prep.get("cached")),
                                                       "blocked": bool(prep.get("blocked"))}).encode()
                else:
                    async for ev in self._stream_generation(req, prep, trace_id):
                        if ev.startswith("__ANSWER__"):
                            answer = ev.removeprefix("__ANSWER__")
                            continue
                        yield ev
            except Exception as exc:  # noqa: BLE001
                log_event(logger, "chat stream failed", logging.ERROR, error=f"{type(exc).__name__}: {exc}")
                yield SSEEvent(event="error", data={"message": UNAVAILABLE}).encode()
                prep.setdefault("reason", "error")
            self._root_update(root, req, prep, answer, t0)

    async def _stream_generation(self, req: ChatRequest, prep: dict[str, Any], trace_id: str) -> AsyncIterator[str]:
        """Relay generator tokens, but pass them through the output guardrail one
        sentence at a time so PII can never reach the customer, even mid-stream."""
        greq = GenerateRequest(query=prep["query"], rewritten_query=prep["rewritten"].rewritten_query,
                               chunks=prep["retrieved"].chunks, history=req.history, stream=True, trace_id=trace_id)
        buf, released, citations, failed, blocked = "", [], [], False, False

        async def flush(text: str) -> AsyncIterator[str]:
            nonlocal blocked
            if not text.strip() or blocked:
                return
            g = await self._guard(text, "output", prep["hdr"])
            if g.blocked:
                blocked = True
                yield SSEEvent(event="token", data={"t": "\n\n" + g.sanitized_text, "replace": True}).encode()
                return
            released.append(g.sanitized_text)
            yield SSEEvent(event="token", data={"t": g.sanitized_text}).encode()

        async for event, data in self.generator.stream_sse("/generate/stream", greq, prep["hdr"]):
            if event == "token":
                buf += json.loads(data)["t"]
                # release everything up to the last sentence boundary
                last = None
                for m in SENTENCE_END.finditer(buf):
                    last = m.end()
                if last:
                    seg, buf = buf[:last], buf[last:]
                    async for ev in flush(seg):
                        yield ev
            elif event == "citations":
                citations = [Citation(**c) for c in json.loads(data)["citations"]]
            elif event == "error":
                failed = True
                yield SSEEvent(event="error", data=json.loads(data)).encode()
            elif event == "done":
                async for ev in flush(buf):
                    yield ev
                buf = ""
                yield SSEEvent(event="citations", data={"citations": [c.model_dump() for c in citations]}).encode()
                yield SSEEvent(event="done", data={**json.loads(data), "cached": False,
                                                   "blocked": blocked}).encode()
        answer = "".join(released)
        await self._maybe_cache(prep, answer, citations, ok=not (failed or blocked))
        yield "__ANSWER__" + answer
