"""End-user / staff authentication (OAuth2 bearer JWT).

In production the bank's IdP (e.g. Keycloak / Okta / Cloud Identity Platform) issues
RS256 tokens; we verify issuer, audience, expiry and required scopes. HS256 with a
shared secret is used for local development and tests.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from bankcore.config import Settings, get_settings
from bankcore.errors import AuthenticationError, AuthorizationError
from bankcore.schemas.common import Role

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    subject: str                     # USR-xxxx for customers, staff id for staff
    role: Role
    scopes: frozenset[str] = field(default_factory=frozenset)
    token_id: str = ""
    mfa_verified: bool = False


def issue_token(subject: str, role: Role, scopes: list[str] | None = None, ttl_s: int = 900,
                mfa: bool = True, settings: Settings | None = None) -> str:
    """Dev/test helper. Production tokens come from the IdP."""
    s = settings or get_settings()
    now = int(time.time())
    claims = {"sub": subject, "role": role.value, "scope": " ".join(scopes or []), "iss": s.jwt_issuer,
              "aud": s.jwt_audience, "iat": now, "nbf": now, "exp": now + ttl_s, "jti": uuid.uuid4().hex,
              "amr": ["pwd", "otp"] if mfa else ["pwd"]}
    return jwt.encode(claims, s.jwt_secret.get_secret_value(), algorithm=s.jwt_algorithm)


def decode_token(token: str, settings: Settings | None = None) -> Principal:
    s = settings or get_settings()
    try:
        claims = jwt.decode(token, s.jwt_secret.get_secret_value(), algorithms=[s.jwt_algorithm],
                            audience=s.jwt_audience, issuer=s.jwt_issuer,
                            options={"require": ["exp", "iat", "sub", "aud", "iss"]}, leeway=10)
    except jwt.ExpiredSignatureError as e:
        raise AuthenticationError("token expired", user_message="Your session has expired.") from e
    except jwt.PyJWTError as e:
        raise AuthenticationError(f"invalid token: {e}") from e
    try:
        role = Role(claims.get("role", "customer"))
    except ValueError as e:
        raise AuthenticationError("unknown role") from e
    return Principal(subject=claims["sub"], role=role, scopes=frozenset(claims.get("scope", "").split()),
                     token_id=claims.get("jti", ""), mfa_verified="otp" in claims.get("amr", []))


async def get_principal(request: Request,
                        creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Principal:
    if creds is None or creds.scheme.lower() != "bearer":
        raise AuthenticationError("missing bearer token")
    principal = decode_token(creds.credentials)
    request.state.principal = principal
    return principal


async def require_customer(p: Principal = Depends(get_principal)) -> Principal:
    if p.role != Role.CUSTOMER:
        raise AuthorizationError("customer role required")
    return p


async def require_staff(p: Principal = Depends(get_principal)) -> Principal:
    if p.role != Role.STAFF or "hitl:approve" not in p.scopes:
        raise AuthorizationError("staff approver role required")
    if not p.mfa_verified:
        raise AuthorizationError("approvers must authenticate with MFA")
    return p
