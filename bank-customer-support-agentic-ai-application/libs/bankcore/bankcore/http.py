"""Shared FastAPI hardening: request ids, security headers, body-size limit, latency metrics,
uniform error responses (no stack traces leak), health and metrics endpoints."""
from __future__ import annotations

import time
import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from bankcore.errors import BankAgentError
from bankcore.logging import get_logger, request_id_var
from bankcore.observability.metrics import REQUEST_LATENCY

log = get_logger(__name__)
MAX_BODY_BYTES = 256 * 1024


def harden(app: FastAPI, service: str) -> FastAPI:
    @app.middleware("http")
    async def _middleware(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request_id_var.set(rid)
        structlog.contextvars.bind_contextvars(request_id=rid)
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
            return JSONResponse({"error": {"code": "payload_too_large", "message": "Request too large."}},
                                status_code=413)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            route = request.scope.get("route")
            REQUEST_LATENCY.labels(service, getattr(route, "path", request.url.path), str(status)).observe(
                time.perf_counter() - started)
        response.headers["x-request-id"] = rid
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["x-frame-options"] = "DENY"
        response.headers["referrer-policy"] = "no-referrer"
        response.headers["cache-control"] = "no-store"
        response.headers["strict-transport-security"] = "max-age=63072000; includeSubDomains"
        return response

    @app.exception_handler(BankAgentError)
    async def _bank_error(_request: Request, exc: BankAgentError):
        if exc.http_status >= 500:
            log.error("request failed", code=exc.code, error=str(exc))
        return JSONResponse({"error": exc.to_dict()}, status_code=exc.http_status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError):
        return JSONResponse({"error": {"code": "validation_failed", "message": "Invalid request.",
                                       "fields": [".".join(map(str, e["loc"])) for e in exc.errors()[:10]]}},
                            status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception):
        log.exception("unhandled error", error=str(exc))
        return JSONResponse({"error": {"code": "internal_error",
                                       "message": "Something went wrong on our side."}}, status_code=500)

    @app.get("/metrics", include_in_schema=False)
    async def _metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
