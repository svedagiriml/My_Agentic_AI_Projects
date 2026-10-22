"""Guardrail microservice.

POST /check        {text, direction: input|output} -> GuardrailResponse
POST /check/batch  [ ... ]                          -> [GuardrailResponse]  (used for streamed segments)
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header

from common.app_factory import create_app
from common.config import get_settings
from common.observability import PARENT_HEADER, TRACE_HEADER, Tracer
from common.schemas import GuardrailRequest, GuardrailResponse
from services.guardrail.engine import GuardrailEngine
from services.guardrail.injection import InjectionDetector
from services.guardrail.pii import PIIRedactor

settings = get_settings()
state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    redactor = await asyncio.to_thread(PIIRedactor, settings.use_presidio)
    detector = await asyncio.to_thread(InjectionDetector, settings.use_injection_classifier,
                                       settings.injection_model, settings.injection_threshold)
    state["engine"] = GuardrailEngine(redactor, detector)
    state["tracer"] = Tracer(settings, "guardrail")
    yield
    state["tracer"].flush()  # type: ignore[attr-defined]


app = create_app("guardrail", settings, lifespan)


@app.post("/check", response_model=GuardrailResponse)
async def check(req: GuardrailRequest, x_trace_id: str | None = Header(None, alias=TRACE_HEADER),
                x_parent: str | None = Header(None, alias=PARENT_HEADER)) -> GuardrailResponse:
    engine: GuardrailEngine = state["engine"]  # type: ignore[assignment]
    tracer: Tracer = state["tracer"]  # type: ignore[assignment]
    resp = await asyncio.to_thread(engine.check, req)   # ML classifier / Presidio are CPU-bound
    # Never send raw PII to the tracing backend: log only the sanitized text + entity types.
    with tracer.span(f"guardrail.{req.direction}", trace_id=x_trace_id, parent_span_id=x_parent,
                     input={"length": len(req.text)}) as span:
        span.update(output={"sanitized_text": resp.sanitized_text, "blocked": resp.blocked,
                            "pii_types": sorted({e.entity_type for e in resp.pii_found}),
                            "injection_score": resp.injection_score, "injection_reasons": resp.injection_reasons})
    return resp


@app.post("/check/batch", response_model=list[GuardrailResponse])
async def check_batch(reqs: list[GuardrailRequest]) -> list[GuardrailResponse]:
    engine: GuardrailEngine = state["engine"]  # type: ignore[assignment]
    return await asyncio.to_thread(lambda: [engine.check(r) for r in reqs])
