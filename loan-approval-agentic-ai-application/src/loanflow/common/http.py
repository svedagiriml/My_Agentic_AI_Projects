"""FastAPI plumbing shared by every service: error mapping, health, metrics, security headers."""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from loanflow.common.exceptions import LoanFlowError
from loanflow.common.logging import get_logger
from loanflow.observability.metrics import REQUESTS, metrics_payload
from loanflow.resilience import all_breakers

log = get_logger(__name__)

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
}


def install_common(app: FastAPI, service: str, max_body_bytes: int = 1_000_000) -> None:
    @app.exception_handler(LoanFlowError)
    async def _lf_error(request: Request, exc: LoanFlowError):
        headers = {}
        if exc.http_status == 429:
            headers["Retry-After"] = str(exc.details.get("retry_after_s", 60))
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()}, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _val_error(request: Request, exc: RequestValidationError):
        errs = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()[:10]]
        return JSONResponse(status_code=422, content={"error": {"code": "INVALID_REQUEST", "message": "request validation failed",
                                                                "retryable": False, "details": {"errors": errs}}})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_ERROR", "message": "internal error",
                                                                "retryable": True, "details": {}}})

    @app.middleware("http")
    async def _mw(request: Request, call_next):
        if int(request.headers.get("content-length") or 0) > max_body_bytes:
            return JSONResponse(status_code=413, content={"error": {"code": "PAYLOAD_TOO_LARGE", "message": "body too large"}})
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        t0 = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        REQUESTS.labels(service, getattr(route, "path", request.url.path), str(response.status_code)).inc()
        response.headers["x-request-id"] = rid
        response.headers["x-response-time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {"status": "ok", "service": service}

    @app.get("/readyz", include_in_schema=False)
    async def readyz():
        from loanflow.memory.kv import get_kv

        kv_ok = await get_kv().ping()
        return JSONResponse(status_code=200 if kv_ok else 503,
                            content={"status": "ready" if kv_ok else "degraded", "redis": kv_ok, "breakers": all_breakers()})

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        body, ctype = metrics_payload()
        return Response(content=body, media_type=ctype)
