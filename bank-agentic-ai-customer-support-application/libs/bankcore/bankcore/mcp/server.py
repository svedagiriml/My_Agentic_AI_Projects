"""Hardened MCP server base used by all eight domain MCP servers.

Every tool registered through ``BankMCPServer.tool`` automatically gets:
  * delegation-token verification (customer identity comes from a signed token, not the LLM)
  * pydantic input validation and output validation (ToolEnvelope + per-tool output model)
  * HITL enforcement for high-risk calls (argument-bound, single-use approval tokens)
  * idempotency keys for money-moving tools
  * per-customer, per-tool rate limits
  * uniform error envelopes (no stack traces / internals cross the boundary)
  * Prometheus metrics, structured audit events

The HTTP app adds service-to-service authentication: only the sub-agent that owns this MCP
domain may call it (network-level state isolation on top of the agent-side allow-list).
"""
from __future__ import annotations

import hashlib
import inspect
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ValidationError
from starlette.responses import Response

from bankcore.agents.policies import approval_requirement
from bankcore.config import Settings, get_settings
from bankcore.errors import AuthenticationError, AuthorizationError, BankAgentError
from bankcore.logging import configure_logging, get_logger
from bankcore.mcp import compat
from bankcore.observability.metrics import TOOL_CALLS, TOOL_LATENCY
from bankcore.resilience.rate_limiter import RateLimiter
from bankcore.schemas.messages import canonical_json
from bankcore.schemas.tools import ToolEnvelope, ToolError
from bankcore.security.service_auth import verify_service_request
from bankcore.security.signing import verify_approval_token, verify_delegation_token

log = get_logger(__name__)
HIDDEN_PARAMS = ("delegation_token", "approval_token")


@dataclass
class ToolContext:
    user_id: str
    agent: str
    thread_id: str
    tool: str
    idempotency_key: str
    approval_claims: dict | None = None


class BankMCPServer:
    def __init__(self, domain: str, instructions: str, *, settings: Settings | None = None,
                 rate_limiter: RateLimiter | None = None, tool_rate_limits: dict[str, tuple[int, int]] | None = None):
        self.domain = domain
        self.settings = settings or get_settings()
        self.server = compat.new_server(f"mcp-{domain}", instructions)
        self.rate_limiter = rate_limiter or RateLimiter(None)
        self.tool_rate_limits = tool_rate_limits or {}
        self._used_approvals: set[str] = set()      # single-use approval jti (Redis SETNX in prod)
        self.tools: dict[str, Callable] = {}

    # ------------------------------------------------------------------ tool --
    def tool(self, *, name: str, description: str, input_model: type[BaseModel] | None,
             output_model: type[BaseModel], high_risk: bool = False):
        def decorator(fn: Callable[..., Awaitable[dict]]):
            sig = inspect.signature(fn)
            params = [p for p in sig.parameters.values() if p.name != "ctx"]
            extra = [inspect.Parameter("delegation_token", inspect.Parameter.KEYWORD_ONLY, annotation=str)]
            if high_risk:
                extra.append(inspect.Parameter("approval_token", inspect.Parameter.KEYWORD_ONLY, annotation=str,
                                               default=""))
            new_params = [p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in params] + extra
            # keyword-only params with defaults must come after... order: required first
            new_params.sort(key=lambda p: p.default is not inspect.Parameter.empty)

            async def wrapper(**kwargs: Any) -> dict[str, Any]:
                return await self._execute(name, fn, input_model, output_model, high_risk, kwargs)

            wrapper.__name__ = name
            wrapper.__doc__ = description
            wrapper.__signature__ = sig.replace(parameters=new_params, return_annotation=dict[str, Any])
            wrapper.__annotations__ = {p.name: p.annotation for p in new_params} | {"return": dict[str, Any]}
            self.server.tool(name=name, description=description)(wrapper)
            self.tools[name] = wrapper
            return wrapper
        return decorator

    async def _execute(self, name: str, fn: Callable, input_model: type[BaseModel] | None,
                       output_model: type[BaseModel], high_risk: bool, kwargs: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        delegation = kwargs.pop("delegation_token", "")
        approval = kwargs.pop("approval_token", "") or ""
        outcome = "ok"
        try:
            claims = verify_delegation_token(delegation, mcp_domain=self.domain, settings=self.settings)
            user_id = claims["sub"]
            if name in self.tool_rate_limits:
                limit, window = self.tool_rate_limits[name]
                await self.rate_limiter.hit(f"tool:{name}:{user_id}", limit=limit, window_s=window)
            validated = input_model.model_validate(kwargs) if input_model else None
            call_args = validated.model_dump() if validated else {}
            json_args = validated.model_dump(mode="json") if validated else {}

            approval_claims = None
            req = approval_requirement(name, json_args, self.settings)
            if high_risk and req.required:
                if not approval:
                    outcome = "approval_required"
                    return ToolEnvelope(ok=False, tool=name, requires_approval=True,
                                        error=ToolError(code="approval_required",
                                                        message=f"Human approval required: {req.reason}")
                                        ).model_dump(mode="json")
                approval_claims = verify_approval_token(approval, tool=name, arguments=json_args, user_id=user_id,
                                                        settings=self.settings)
                if approval_claims["jti"] in self._used_approvals:
                    raise AuthorizationError("approval token already used")
                self._used_approvals.add(approval_claims["jti"])

            idem_src = approval_claims["aid"] if approval_claims else f"{claims.get('thr')}|{name}|" \
                                                                         f"{canonical_json(json_args)}"
            ctx = ToolContext(user_id=user_id, agent=claims["act"], thread_id=claims.get("thr", ""), tool=name,
                              idempotency_key=hashlib.sha256(idem_src.encode()).hexdigest()[:32],
                              approval_claims=approval_claims)
            raw = await fn(ctx, **call_args)
            data = output_model.model_validate(raw).model_dump(mode="json")      # output verification
            log.info("tool executed", tool=name, agent=ctx.agent, user=user_id, approved=bool(approval_claims))
            return ToolEnvelope(ok=True, tool=name, data=data).model_dump(mode="json")
        except ValidationError as e:
            outcome = "invalid_input"
            msg = "; ".join(f"{'.'.join(map(str, er['loc']))}: {er['msg']}" for er in e.errors()[:5])
            return ToolEnvelope(ok=False, tool=name, error=ToolError(code="invalid_arguments", message=msg)
                                ).model_dump(mode="json")
        except (AuthenticationError, AuthorizationError) as e:
            outcome = "denied"
            log.warning("tool call denied", tool=name, reason=str(e))
            return ToolEnvelope(ok=False, tool=name, error=ToolError(code=e.code, message=e.user_message)
                                ).model_dump(mode="json")
        except BankAgentError as e:
            outcome = e.code
            return ToolEnvelope(ok=False, tool=name, error=ToolError(code=e.code, message=e.user_message)
                                ).model_dump(mode="json")
        except Exception as e:  # noqa: BLE001
            outcome = "internal_error"
            log.exception("tool crashed", tool=name, error=str(e))
            return ToolEnvelope(ok=False, tool=name, error=ToolError(
                code="internal_error", message="The service could not complete this request.")).model_dump(mode="json")
        finally:
            TOOL_CALLS.labels(self.domain, name, outcome).inc()
            TOOL_LATENCY.labels(self.domain, name).observe(time.perf_counter() - started)

    # ------------------------------------------------------------------- app --
    def build_app(self) -> FastAPI:
        configure_logging(f"mcp-{self.domain}", self.settings.log_level)
        mcp_app = compat.http_app(self.server)
        allowed = {f"agent-{self.domain}"}

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            async with compat.session_manager(self.server).run():
                yield

        app = FastAPI(title=f"mcp-{self.domain}", lifespan=lifespan, docs_url=None, redoc_url=None)

        @app.middleware("http")
        async def service_auth(request: Request, call_next):
            if request.url.path.startswith("/mcp"):
                try:
                    verify_service_request(request, audience=f"mcp-{self.domain}", allowed_callers=allowed,
                                           settings=self.settings)
                except BankAgentError as e:
                    return JSONResponse({"error": e.code}, status_code=e.http_status)
            return await call_next(request)

        @app.get("/healthz")
        async def healthz():
            return {"status": "ok", "domain": self.domain, "tools": sorted(self.tools)}

        @app.get("/metrics")
        async def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        app.mount("/mcp", mcp_app)
        return app
