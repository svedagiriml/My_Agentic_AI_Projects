"""Generator microservice (FastAPI, async, streaming).

POST /rewrite          raw query -> RewrittenQuery (Pydantic structured output)
POST /generate         grounded answer -> GenerateResponse (Pydantic structured output)
POST /generate/stream  grounded answer as Server-Sent Events (token / citations / done)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from common.app_factory import create_app
from common.config import get_settings
from common.observability import PARENT_HEADER, TRACE_HEADER, Tracer
from common.schemas import GenerateRequest, GenerateResponse, RewriteRequest, RewrittenQuery
from services.generator.llm_client import LLMClient
from services.generator.service import GeneratorService

settings = get_settings()
state: dict[str, GeneratorService] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    tracer = Tracer(settings, "generator")
    state["svc"] = GeneratorService(settings, tracer, LLMClient(settings, "answer"), LLMClient(settings, "rewrite"))
    yield
    tracer.flush()


app = create_app("generator", settings, lifespan)


def _svc() -> GeneratorService:
    if "svc" not in state:
        raise HTTPException(503, "generator warming up")
    return state["svc"]


@app.post("/rewrite", response_model=RewrittenQuery)
async def rewrite(req: RewriteRequest, x_trace_id: str | None = Header(None, alias=TRACE_HEADER),
                  x_parent: str | None = Header(None, alias=PARENT_HEADER)) -> RewrittenQuery:
    req.trace_id = req.trace_id or x_trace_id
    try:
        return await _svc().rewrite(req, x_parent)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, x_trace_id: str | None = Header(None, alias=TRACE_HEADER),
                   x_parent: str | None = Header(None, alias=PARENT_HEADER)) -> GenerateResponse:
    req.trace_id = req.trace_id or x_trace_id
    try:
        return await _svc().generate(req, x_parent)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest, x_trace_id: str | None = Header(None, alias=TRACE_HEADER),
                          x_parent: str | None = Header(None, alias=PARENT_HEADER)) -> StreamingResponse:
    req.trace_id = req.trace_id or x_trace_id
    return StreamingResponse(_svc().generate_stream(req, x_parent), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
