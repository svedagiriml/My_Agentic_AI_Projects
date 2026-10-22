"""FastAPI app factory: logging, trace propagation, latency metrics, throttling, health."""
from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from common.config import Settings
from common.log import setup_logging, trace_id_var
from common.metrics import HTTP_INFLIGHT, HTTP_LATENCY, THROTTLED, metrics_payload
from common.observability import TRACE_HEADER
from common.resilience import AsyncTokenBucket


class ClientThrottle:
    """Per-client token buckets (LRU-bounded so memory can't grow unbounded)."""

    def __init__(self, per_minute: int, max_clients: int = 10000) -> None:
        self.per_minute = per_minute
        self.max_clients = max_clients
        self._buckets: OrderedDict[str, AsyncTokenBucket] = OrderedDict()

    def allow(self, key: str) -> bool:
        b = self._buckets.get(key)
        if b is None:
            b = AsyncTokenBucket.per_minute(self.per_minute)
            self._buckets[key] = b
            if len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
        return b.try_acquire()


def create_app(service: str, settings: Settings,
               lifespan: Callable[[FastAPI], AbstractAsyncContextManager[Any]] | None = None,
               throttle_per_minute: int | None = None) -> FastAPI:
    setup_logging(service, settings.log_level)
    app = FastAPI(title=f"bank-rag-{service}", version="1.0.0", lifespan=lifespan)
    throttle = ClientThrottle(throttle_per_minute) if throttle_per_minute else None
    open_paths = {"/healthz", "/readyz", "/metrics"}

    @app.middleware("http")
    async def _observe(request: Request, call_next):  # type: ignore[no-untyped-def]
        token = trace_id_var.set(request.headers.get(TRACE_HEADER))
        route = request.url.path
        if throttle and route not in open_paths:
            client = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
            if not throttle.allow(client.split(",")[0].strip()):
                THROTTLED.labels(service).inc()
                trace_id_var.reset(token)
                return JSONResponse({"detail": "rate limit exceeded, slow down"}, status_code=429,
                                    headers={"Retry-After": "5"})
        HTTP_INFLIGHT.labels(service).inc()
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            HTTP_INFLIGHT.labels(service).dec()
            HTTP_LATENCY.labels(service, route, request.method, str(status)).observe(time.perf_counter() - start)
            trace_id_var.reset(token)

    @app.get("/healthz", include_in_schema=False)
    async def _health() -> dict[str, str]:
        return {"status": "ok", "service": service}

    @app.get("/metrics", include_in_schema=False)
    async def _metrics() -> Response:
        body, ctype = metrics_payload()
        return Response(body, media_type=ctype)

    return app
