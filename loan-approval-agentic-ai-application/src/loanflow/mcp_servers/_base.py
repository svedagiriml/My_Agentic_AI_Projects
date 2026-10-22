"""Shared plumbing for MCP servers: fixtures, auth middleware, health endpoint.

In production each MCP server is a private Cloud Run service (ingress=internal,
no unauthenticated invocations). Defence in depth: we *also* require the internal
service token (or a Google-signed ID token when USE_GCP_ID_TOKENS=true).
"""
from __future__ import annotations

import hmac
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from loanflow.common.config import get_settings
from loanflow.mcp_layer.compat import http_app

DATA = Path(__file__).parent / "data" / "fixtures.json"


@lru_cache(maxsize=1)
def fixtures() -> dict[str, Any]:
    return json.loads(DATA.read_text())


def _verify_google_id_token(token: str) -> bool:
    try:
        from google.auth.transport import requests as greq
        from google.oauth2 import id_token

        id_token.verify_oauth2_token(token, greq.Request())
        return True
    except Exception:  # noqa: BLE001
        return False


class InternalAuthMiddleware:
    """ASGI middleware: rejects MCP calls without a valid internal bearer token."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path", "") in ("/healthz", "/readyz"):
            if scope["type"] == "http" and scope.get("path") in ("/healthz", "/readyz"):
                await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"status":"ok"}'})
                return
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        token = headers.get("authorization", "").removeprefix("Bearer ").strip()
        s = get_settings()
        ok = hmac.compare_digest(token, s.internal_service_token) or (s.use_gcp_id_tokens and _verify_google_id_token(token))
        if not ok:
            await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
            return
        return await self.app(scope, receive, send)


def secure_asgi(server: Any) -> Any:
    return InternalAuthMiddleware(http_app(server))


def tool_error(code: str, message: str) -> dict[str, Any]:
    """Business-level tool failure returned as data (not an exception) so the agent can reason about it."""
    return {"ok": False, "error": {"code": code, "message": message}}
