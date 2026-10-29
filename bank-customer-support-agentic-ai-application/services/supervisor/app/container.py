"""Dependency wiring for the supervisor (built once per instance at startup).

Every external dependency has a degraded-mode fallback so the API keeps serving:
  Redis down     -> in-process short-term memory, local rate limiter, no routing cache
  Postgres down  -> in-memory checkpointer (approvals then don't survive restarts: alerted) and no LTM
  LLM down       -> keyword router + direct answers
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bankcore.agents.specs import PROMPT_VERSION
from bankcore.cache.semantic_cache import SemanticRoutingCache
from bankcore.config import Settings
from bankcore.guardrails import InputGuard
from bankcore.llm.embeddings import get_embedder
from bankcore.llm.factory import ResilientLLM
from bankcore.logging import get_logger
from bankcore.memory.long_term import LongTermMemory, PgVectorLongTermMemory
from bankcore.memory.manager import MemoryManager
from bankcore.memory.short_term import InMemoryShortTermMemory, ShortTermMemory
from bankcore.messaging.event_bus import EventBus, InMemoryBus, build_event_bus
from bankcore.resilience.rate_limiter import RateLimiter

from supervisor.app.agent_client import HttpSubAgentClient, SubAgentClient
from supervisor.app.graph import SupervisorDeps, build_supervisor_graph
from supervisor.app.hitl import HitlRegistry

log = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    graph: Any
    deps: SupervisorDeps
    rate_limiter: RateLimiter
    redis: Any = None
    long_term: LongTermMemory | None = None
    _pg_ctx: Any = None

    async def aclose(self) -> None:
        if self._pg_ctx is not None:
            await self._pg_ctx.__aexit__(None, None, None)
        if self.redis is not None:
            await self.redis.aclose()


async def _redis(settings: Settings):
    try:
        from redis.asyncio import Redis
        r = Redis.from_url(settings.redis_url, socket_timeout=2.0, socket_connect_timeout=2.0,
                           health_check_interval=30)
        await r.ping()
        return r
    except Exception as e:  # noqa: BLE001
        log.error("redis unavailable — degraded mode", error=str(e))
        return None


async def _checkpointer(settings: Settings):
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        ctx = AsyncPostgresSaver.from_conn_string(settings.postgres_dsn.get_secret_value())
        saver = await ctx.__aenter__()
        await saver.setup()
        return saver, ctx
    except Exception as e:  # noqa: BLE001
        log.error("postgres checkpointer unavailable — using in-memory (NOT durable)", error=str(e))
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver(), None


async def build_container(settings: Settings, *, agents: SubAgentClient | None = None, llm: ResilientLLM | None = None,
                          checkpointer=None, use_external: bool = True) -> Container:
    redis = await _redis(settings) if use_external else None
    st: ShortTermMemory = ShortTermMemory(redis, settings.short_term_ttl_s, settings.short_term_max_turns) if redis \
        else InMemoryShortTermMemory(settings.short_term_max_turns)
    lt: LongTermMemory | None = None
    pg_ctx = None
    if use_external:
        try:
            lt = await PgVectorLongTermMemory.create(settings.postgres_dsn.get_secret_value(), get_embedder())
        except Exception as e:  # noqa: BLE001
            log.error("long-term memory unavailable", error=str(e))
        if checkpointer is None:
            checkpointer, pg_ctx = await _checkpointer(settings)
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    bus: EventBus = build_event_bus(settings, redis) if use_external else InMemoryBus()
    cache = SemanticRoutingCache(redis, get_embedder(), threshold=settings.semantic_cache_threshold,
                                 ttl_s=settings.semantic_cache_ttl_s, version=PROMPT_VERSION) \
        if (redis is not None and settings.semantic_cache_enabled) else None
    if llm is None:
        try:
            llm = ResilientLLM("supervisor", settings)
        except Exception as e:  # noqa: BLE001
            log.error("no supervisor LLM — keyword routing only", error=str(e))
    deps = SupervisorDeps(settings=settings, llm=llm, agents=agents or HttpSubAgentClient(settings),
                          memory=MemoryManager(st, lt), bus=bus, hitl=HitlRegistry(redis), cache=cache,
                          input_guard=InputGuard())
    graph = build_supervisor_graph(deps, checkpointer)
    return Container(settings=settings, graph=graph, deps=deps, rate_limiter=RateLimiter(redis), redis=redis,
                     long_term=lt, _pg_ctx=pg_ctx)
