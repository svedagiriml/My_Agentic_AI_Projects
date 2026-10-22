"""Public API gateway (the only internet-facing Cloud Run service).

POST /v1/chat          JSON answer (stream=false) or SSE stream (stream=true, default)
GET  /                 minimal chat UI (static) for demos
Per-client throttling (token bucket per IP) protects the free LLM quota from a
single noisy client; all work is async so one worker serves many concurrent
streams while waiting on upstream I/O.
"""
from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from common.app_factory import create_app
from common.config import get_settings
from common.http_client import ServiceClient
from common.observability import Tracer
from common.schemas import ChatRequest, ChatResponse
from services.gateway.orchestrator import Orchestrator
from services.gateway.semantic_cache import InMemorySemanticCache, RedisSemanticCache

settings = get_settings()
state: dict[str, object] = {}
STATIC = pathlib.Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    kw = dict(timeout_s=settings.internal_timeout_s, use_id_token=settings.use_gcp_id_token,
              attempts=settings.retry_max_attempts, base_delay=settings.retry_base_delay_s,
              max_delay=settings.retry_max_delay_s, breaker_threshold=settings.breaker_failure_threshold,
              breaker_reset_s=settings.breaker_reset_timeout_s)
    clients = {n: ServiceClient(n, str(u), **kw) for n, u in (("retriever", settings.retriever_url),
                                                              ("generator", settings.generator_url),
                                                              ("guardrail", settings.guardrail_url))}
    cache = (RedisSemanticCache(settings.redis_url, settings.embedding_dim, settings.cache_similarity_threshold,
                                settings.cache_ttl_s)
             if settings.cache_backend == "redis"
             else InMemorySemanticCache(settings.cache_similarity_threshold, settings.cache_ttl_s,
                                        settings.cache_max_entries))
    tracer = Tracer(settings, "gateway")
    state["orch"] = Orchestrator(settings, tracer, cache, **clients)
    yield
    for c in clients.values():
        await c.aclose()
    tracer.flush()


app = create_app("gateway", settings, lifespan, throttle_per_minute=settings.api_rate_limit_per_minute)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):  # type: ignore[no-untyped-def]
    orch: Orchestrator = state["orch"]  # type: ignore[assignment]
    if req.stream:
        return StreamingResponse(orch.chat_stream(req), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    return await orch.chat(req)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
