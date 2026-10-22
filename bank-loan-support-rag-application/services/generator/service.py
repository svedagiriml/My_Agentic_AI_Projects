"""Generator core: query rewriting + grounded answer generation (structured or streamed)."""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from common.config import Settings
from common.cost import compute_cost
from common.observability import Tracer
from common.schemas import (
    CITATION_RE,
    ChatTurn,
    Citation,
    GeneratedAnswer,
    GenerateRequest,
    GenerateResponse,
    RetrievedChunk,
    RewriteRequest,
    RewrittenQuery,
    SSEEvent,
    TokenUsage,
)
from services.generator.llm_client import LLMClient
from services.generator.prompts import ANSWER_PROMPT, REWRITE_PROMPT, STREAM_FORMAT, STRUCTURED_FORMAT, format_context


def _history(turns: list[ChatTurn]) -> list[BaseMessage]:
    return [HumanMessage(t.content) if t.role == "user" else AIMessage(t.content) for t in turns[-6:]]


def citations_from_labels(labels: list[str], label_map: dict[str, RetrievedChunk]) -> list[Citation]:
    out, seen = [], set()
    for lab in labels:
        c = label_map.get(lab)
        if c and c.chunk_id not in seen:
            seen.add(c.chunk_id)
            out.append(Citation(chunk_id=c.chunk_id, source_file=c.source_file, page=c.page))
    return out


def labels_in_text(text: str) -> list[str]:
    return [f"S{n}" for n in CITATION_RE.findall(text)]


class GeneratorService:
    def __init__(self, settings: Settings, tracer: Tracer, answer_llm: LLMClient, rewrite_llm: LLMClient) -> None:
        self.s = settings
        self.tracer = tracer
        self.answer_llm = answer_llm
        self.rewrite_llm = rewrite_llm

    # ------------------------------------------------------------------ rewrite
    async def rewrite(self, req: RewriteRequest, parent_span_id: str | None = None) -> RewrittenQuery:
        messages = REWRITE_PROMPT.format_messages(history=_history(req.history), question=req.query)
        with self.tracer.generation("generator.rewrite", trace_id=req.trace_id, parent_span_id=parent_span_id,
                                    model=self.rewrite_llm.primary_model,
                                    input=[{"role": m.type, "content": m.content} for m in messages]) as gen:
            parsed, usage, model = await self.rewrite_llm.structured(messages, RewrittenQuery, "rewrite")
            cost, shadow = compute_cost(model, usage, self.s)
            gen.update(model=model, output=parsed.model_dump(mode="json"),
                       usage_details={"input": usage.input_tokens, "output": usage.output_tokens},
                       cost_details={"total": cost}, metadata={"shadow_cost_usd": shadow,
                                                              "raw_query": req.query})
        return parsed

    # ------------------------------------------------------------------ helpers
    def _messages(self, req: GenerateRequest, fmt: str) -> tuple[list[BaseMessage], dict[str, RetrievedChunk], str]:
        context, label_map = format_context(req.chunks)
        msgs = ANSWER_PROMPT.format_messages(history=_history(req.history), context=context,
                                             question=req.query, format_instructions=fmt)
        return msgs, label_map, context

    def _trace_input(self, req: GenerateRequest, msgs: list[BaseMessage],
                     label_map: dict[str, RetrievedChunk]) -> dict[str, Any]:
        # Full prompt composition for LangFuse: every message + which chunk sits behind each label.
        return {
            "messages": [{"role": m.type, "content": m.content} for m in msgs],
            "composition": {
                "raw_query": req.query, "rewritten_query": req.rewritten_query,
                "context_passages": [{"label": k, "chunk_id": v.chunk_id, "rrf_score": v.rrf_score,
                                      "dense_score": v.dense_score, "bm25_score": v.bm25_score}
                                     for k, v in label_map.items()],
                "history_turns": len(req.history),
            },
        }

    # ------------------------------------------------------------------ generate
    async def generate(self, req: GenerateRequest, parent_span_id: str | None = None) -> GenerateResponse:
        t0 = time.perf_counter()
        msgs, label_map, _ = self._messages(req, STRUCTURED_FORMAT)
        with self.tracer.generation("generator.generate", trace_id=req.trace_id, parent_span_id=parent_span_id,
                                    model=self.answer_llm.primary_model, input=self._trace_input(req, msgs, label_map),
                                    model_parameters={"temperature": self.s.llm_temperature,
                                                      "max_tokens": self.s.llm_max_output_tokens}) as gen:
            parsed, usage, model = await self.answer_llm.structured(msgs, GeneratedAnswer, "generate")
            labels = parsed.cited_sources or labels_in_text(parsed.answer)
            cost, shadow = compute_cost(model, usage, self.s)
            gen.update(model=model, output=parsed.model_dump(mode="json"),
                       usage_details={"input": usage.input_tokens, "output": usage.output_tokens},
                       cost_details={"total": cost}, metadata={"shadow_cost_usd": shadow})
        return GenerateResponse(
            answer=parsed.answer, citations=citations_from_labels(labels, label_map),
            confidence=parsed.confidence, answerable_from_context=parsed.answerable_from_context,
            needs_human_agent=parsed.needs_human_agent, model=model, usage=usage, cost_usd=cost,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    async def generate_stream(self, req: GenerateRequest, parent_span_id: str | None = None) -> AsyncIterator[str]:
        """SSE stream: token* -> citations -> done (usage/cost). Errors become an `error` event."""
        t0 = time.perf_counter()
        msgs, label_map, _ = self._messages(req, STREAM_FORMAT)
        usage_box: dict[str, Any] = {}
        parts: list[str] = []
        with self.tracer.generation("generator.generate_stream", trace_id=req.trace_id,
                                    parent_span_id=parent_span_id, model=self.answer_llm.primary_model,
                                    input=self._trace_input(req, msgs, label_map)) as gen:
            try:
                async for tok in self.answer_llm.stream(msgs, usage_box):
                    parts.append(tok)
                    yield SSEEvent(event="token", data={"t": tok}).encode()
            except Exception as exc:  # noqa: BLE001
                gen.update(level="ERROR", status_message=str(exc)[:300])
                yield SSEEvent(event="error", data={"message": "The assistant is temporarily unavailable. "
                                                               "Please try again shortly."}).encode()
                return
            answer = "".join(parts)
            usage: TokenUsage = usage_box.get("usage", TokenUsage())
            model = usage_box.get("model", self.answer_llm.primary_model)
            cost, shadow = compute_cost(model, usage, self.s)
            cits = citations_from_labels(labels_in_text(answer), label_map)
            gen.update(model=model, output=answer, usage_details={"input": usage.input_tokens,
                                                                  "output": usage.output_tokens},
                       cost_details={"total": cost}, metadata={"shadow_cost_usd": shadow,
                                                              "citations": [c.chunk_id for c in cits]})
        yield SSEEvent(event="citations", data={"citations": [c.model_dump() for c in cits]}).encode()
        yield SSEEvent(event="done", data={"model": model, "usage": usage.model_dump(), "cost_usd": cost,
                                           "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}).encode()
