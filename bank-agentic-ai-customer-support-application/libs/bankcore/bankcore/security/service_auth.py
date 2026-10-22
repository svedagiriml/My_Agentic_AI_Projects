"""Service-to-service authentication.

* ``gcp``   — Cloud Run IAM: callers attach a Google-signed ID token whose audience is the
  callee URL; Cloud Run rejects unauthenticated calls before they reach the container
  (``--no-allow-unauthenticated``). We additionally verify the caller e-mail allow-list.
* ``local`` — HS256 JWT signed with the handoff secret, audience = callee service name.
"""
from __future__ import annotations

import time

import jwt
from fastapi import Request

from bankcore.config import Settings, get_settings
from bankcore.errors import AuthenticationError, AuthorizationError

_token_cache: dict[str, tuple[str, float]] = {}


def outbound_service_token(audience: str, caller: str, settings: Settings | None = None,
                           url: str | None = None) -> str:
    """``audience`` is the logical callee name (local mode). In gcp mode the Google ID token's
    audience must be the callee's Cloud Run URL, so callers pass ``url`` as well."""
    s = settings or get_settings()
    if s.service_auth_mode == "gcp" and url:
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        audience = f"{parts.scheme}://{parts.netloc}"
    cached = _token_cache.get(audience)
    if cached and cached[1] - 60 > time.time():
        return cached[0]
    if s.service_auth_mode == "gcp":
        import google.auth.transport.requests
        import google.oauth2.id_token

        token = google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), audience)
        exp = time.time() + 3000
    else:
        now = int(time.time())
        exp = now + 600
        token = jwt.encode({"iss": caller, "sub": caller, "aud": audience, "iat": now, "exp": exp},
                           s.handoff_hmac_secret.get_secret_value(), algorithm="HS256")
    _token_cache[audience] = (token, exp)
    return token


def verify_service_request(request: Request, *, audience: str, allowed_callers: set[str],
                           settings: Settings | None = None) -> str:
    """Returns the caller identity or raises."""
    s = settings or get_settings()
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise AuthenticationError("missing service token")
    token = header.split(" ", 1)[1]
    if s.service_auth_mode == "gcp":
        import google.auth.transport.requests
        import google.oauth2.id_token

        import os
        # Cloud Run IAM already verified the token (--no-allow-unauthenticated); we re-verify it
        # against our own URL and enforce the caller allow-list (service-account e-mail prefix).
        try:
            claims = google.oauth2.id_token.verify_oauth2_token(
                token, google.auth.transport.requests.Request(), audience=os.getenv("SELF_URL"))
        except Exception as e:  # noqa: BLE001
            raise AuthenticationError(f"bad id token: {e}") from e
        caller = claims.get("email", "")
        if allowed_callers and not any(caller.startswith(f"{c}@") for c in allowed_callers):
            raise AuthorizationError(f"caller {caller} not allowed")
        return caller
    try:
        claims = jwt.decode(token, s.handoff_hmac_secret.get_secret_value(), algorithms=["HS256"],
                            audience=audience, options={"require": ["exp", "sub", "aud"]})
    except jwt.PyJWTError as e:
        raise AuthenticationError(f"bad service token: {e}") from e
    if allowed_callers and claims["sub"] not in allowed_callers:
        raise AuthorizationError(f"caller {claims['sub']} not allowed to call {audience}")
    return claims["sub"]
