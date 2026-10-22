"""LangFuse observability wrapper (works across microservices).

Distributed tracing model
-------------------------
The gateway mints a W3C-style trace id (32 hex chars) and opens the root span
`chat_request`. Every downstream HTTP call carries two headers:

    X-Trace-Id:        <trace id>
    X-Parent-Span-Id:  <span id of the caller's current span>

Each service opens its own span with `trace_context={trace_id, parent_span_id}`,
so LangFuse stitches one execution tree per user query:

chat_request (gateway)
 ├─ guardrail.input            redacted query, PII types, injection score
 ├─ semantic_cache.lookup      hit/miss, similarity
 ├─ generator.rewrite          raw query -> rewritten query (LLM generation)
 ├─ retriever.hybrid_search    chunk ids, dense/BM25/RRF scores per chunk
 ├─ generator.generate         final prompt composition + answer (LLM generation, tokens, cost)
 └─ guardrail.output           PII scrubbed from the answer

Observations are created with `start_span` / `start_generation` (explicit
`.end()`), not the "as_current" variants, because spans stay open across
`yield`s of streaming responses where OpenTelemetry context-var tokens cannot be
detached safely.

If LangFuse keys are absent (local tests) every call becomes a no-op.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from common.config import Settings

logger = logging.getLogger(__name__)

TRACE_HEADER = "X-Trace-Id"
PARENT_HEADER = "X-Parent-Span-Id"


def new_trace_id() -> str:
    return uuid.uuid4().hex  # 32 lowercase hex == W3C trace id


class _NoopObservation:
    id: str | None = None
    trace_id: str | None = None

    def update(self, **_: Any) -> _NoopObservation:
        return self

    def end(self, **_: Any) -> None:
        return None

    def update_trace(self, **_: Any) -> _NoopObservation:
        return self


class Tracer:
    def __init__(self, settings: Settings, service: str) -> None:
        self.service = service
        self._client = None
        if settings.langfuse_enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
                    host=settings.langfuse_host,
                    environment=settings.environment,
                )
            except Exception:  # noqa: BLE001 - observability must never break serving
                logger.exception("LangFuse init failed; tracing disabled")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @staticmethod
    def _ctx(trace_id: str | None, parent_span_id: str | None) -> dict[str, str] | None:
        if not trace_id:
            return None
        ctx = {"trace_id": trace_id}
        if parent_span_id:
            ctx["parent_span_id"] = parent_span_id
        return ctx

    @contextmanager
    def span(self, name: str, *, trace_id: str | None, parent_span_id: str | None = None,
             input: Any = None, metadata: dict[str, Any] | None = None) -> Iterator[Any]:
        if not self._client:
            yield _NoopObservation()
            return
        try:
            obs = self._client.start_span(
                name=name, trace_context=self._ctx(trace_id, parent_span_id), input=input,
                metadata={"service": self.service, **(metadata or {})},
            )
        except Exception:  # noqa: BLE001
            logger.exception("LangFuse span start failed")
            yield _NoopObservation()
            return
        try:
            yield obs
        except BaseException as exc:
            obs.update(level="ERROR", status_message=str(exc)[:500])
            raise
        finally:
            obs.end()

    @contextmanager
    def generation(self, name: str, *, trace_id: str | None, parent_span_id: str | None = None,
                   model: str, input: Any, model_parameters: dict[str, Any] | None = None,
                   metadata: dict[str, Any] | None = None) -> Iterator[Any]:
        if not self._client:
            yield _NoopObservation()
            return
        try:
            obs = self._client.start_generation(
                name=name, trace_context=self._ctx(trace_id, parent_span_id), model=model,
                input=input, model_parameters=model_parameters or {},
                metadata={"service": self.service, **(metadata or {})},
            )
        except Exception:  # noqa: BLE001
            logger.exception("LangFuse generation start failed")
            yield _NoopObservation()
            return
        try:
            yield obs
        except BaseException as exc:
            obs.update(level="ERROR", status_message=str(exc)[:500])
            raise
        finally:
            obs.end()

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                logger.exception("LangFuse flush failed")


def trace_headers(trace_id: str | None, span: Any) -> dict[str, str]:
    h: dict[str, str] = {}
    if trace_id:
        h[TRACE_HEADER] = trace_id
    sid = getattr(span, "id", None)
    if sid:
        h[PARENT_HEADER] = sid
    return h
