"""Authentication & authorisation.

* End users (loan officers, underwriters) -> JWT bearer tokens (HS256 in dev; in prod
  issue RS256 tokens from your IdP / Identity Platform and set JWT_SECRET to the PEM
  public key + algorithm). Roles drive RBAC:
      loan_officer  : submit applications, read status
      underwriter   : everything above + approve/reject HITL reviews
      admin         : everything + memory quarantine / ops endpoints
* Service-to-service (supervisor -> agents -> MCP) -> Cloud Run IAM (Google-signed
  ID tokens) plus an internal shared token as defence in depth.
"""
from __future__ import annotations

import hmac
import time
from typing import Literal

import jwt
from fastapi import Depends, Header
from pydantic import BaseModel

from loanflow.common.config import get_settings
from loanflow.common.exceptions import AuthError, ForbiddenError

Role = Literal["loan_officer", "underwriter", "admin", "service"]


class Principal(BaseModel):
    sub: str
    roles: list[Role]


def issue_token(sub: str, roles: list[Role], ttl_s: int = 3600) -> str:
    """Dev/test helper (prod tokens come from the IdP)."""
    s = get_settings()
    now = int(time.time())
    return jwt.encode(
        {"sub": sub, "roles": roles, "iss": s.jwt_issuer, "aud": s.jwt_audience, "iat": now, "exp": now + ttl_s},
        s.jwt_secret, algorithm="HS256",
    )


def decode_token(token: str) -> Principal:
    s = get_settings()
    try:
        claims = jwt.decode(token, s.jwt_secret, algorithms=["HS256"], audience=s.jwt_audience, issuer=s.jwt_issuer,
                            options={"require": ["exp", "sub", "iss", "aud"]})
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("token expired") from exc
    except jwt.PyJWTError as exc:
        raise AuthError("invalid token") from exc
    return Principal(sub=claims["sub"], roles=claims.get("roles", []))


async def current_user(authorization: str = Header(default="")) -> Principal:
    if not authorization.startswith("Bearer "):
        raise AuthError("missing bearer token")
    return decode_token(authorization.removeprefix("Bearer ").strip())


def require_roles(*roles: Role):
    async def dep(user: Principal = Depends(current_user)) -> Principal:
        if not set(roles) & set(user.roles) and "admin" not in user.roles:
            raise ForbiddenError(f"requires one of roles {list(roles)}")
        return user

    return dep


async def require_service(authorization: str = Header(default="")) -> Principal:
    """Internal endpoints (agent services). Accepts the internal token or a Google ID token."""
    token = authorization.removeprefix("Bearer ").strip()
    s = get_settings()
    if token and hmac.compare_digest(token, s.internal_service_token):
        return Principal(sub="internal", roles=["service"])
    if s.use_gcp_id_tokens and token:
        try:
            from google.auth.transport import requests as greq
            from google.oauth2 import id_token

            info = id_token.verify_oauth2_token(token, greq.Request())
            return Principal(sub=info.get("email", "gcp-sa"), roles=["service"])
        except Exception as exc:  # noqa: BLE001
            raise AuthError("invalid service identity token") from exc
    raise AuthError("service authentication required")
