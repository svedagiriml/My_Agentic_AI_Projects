"""Memory consolidation & audit worker (asynchronous, off the request path).

Consumes ``memory.consolidate`` and ``audit.events``:
  * episodic   — one summary per conversation turn (user-namespaced, 180-day TTL)
  * semantic   — durable customer preferences stated by the user (pattern-based, optional LLM extractor)
  * procedural — successful, compliant, non-rejected plans become reusable routing know-how (global,
                 entity-masked so no customer data leaks between tenants)
  * audit      — append-only audit_log rows (7-year retention, BigQuery export in prod)
All writes go through MemoryGuard (anti-poisoning). Handlers are idempotent (dedupe by thread_id).

Runs as a Cloud Run service: Redis Streams consumer loop locally, or a Pub/Sub push endpoint
(POST /pubsub/push) in the cloud.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request

from bankcore.cache.semantic_cache import mask_entities
from bankcore.config import get_settings
from bankcore.http import harden
from bankcore.llm.embeddings import get_embedder
from bankcore.logging import configure_logging, get_logger
from bankcore.memory.long_term import InMemoryLongTermMemory, LongTermMemory, PgVectorLongTermMemory
from bankcore.messaging.event_bus import TOPIC_AUDIT, TOPIC_MEMORY, RedisStreamsBus
from bankcore.schemas.memory import MemoryKind, MemoryRecord, MemorySource

log = get_logger(__name__)

_PREF_PATTERNS = [
    re.compile(r"\b(?:i prefer|i'd prefer|i would prefer|please always)\s+(.{3,80}?)(?:[.!?]|$)", re.I),
    re.compile(r"\bmy preferred (language|branch|contact method|channel) is\s+(.{2,40}?)(?:[.!?]|$)", re.I),
    re.compile(r"\bcall me\s+([A-Za-z][A-Za-z .'-]{1,30})(?:[.!?,]|$)", re.I),
]


def extract_preferences(message: str) -> list[str]:
    facts = []
    for p in _PREF_PATTERNS:
        for m in p.finditer(message):
            facts.append("Customer stated: " + m.group(0).strip().rstrip(".!?"))
    return facts[:3]


class Consolidator:
    def __init__(self, ltm: LongTermMemory, redis=None) -> None:
        self.ltm, self.redis = ltm, redis
        self._seen: set[str] = set()

    async def _first_time(self, key: str) -> bool:
        if self.redis is not None:
            try:
                return bool(await self.redis.set(f"processed:{key}", 1, nx=True, ex=7 * 86400))
            except Exception:  # noqa: BLE001
                pass
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    async def handle(self, ev: dict) -> dict:
        if not await self._first_time(f"mem:{ev.get('thread_id')}"):
            return {"skipped": "duplicate"}
        user, trace = ev.get("user_id"), ev.get("trace_id")
        written = {"episodic": 0, "semantic": 0, "procedural": 0}
        if ev.get("status") in ("blocked",):
            return written                       # never learn from blocked / malicious turns
        statuses = ev.get("task_statuses", {})
        agents = sorted({t["agent"] for t in (ev.get("plan") or {}).get("tasks", [])})
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        summary = (f"{when}: customer asked '{ev.get('message', '')[:200]}'. Handled by {', '.join(agents) or 'n/a'}; "
                   f"outcome={ev.get('status')} {statuses}.")
        if await self.ltm.write(MemoryRecord(user_id=user, kind=MemoryKind.EPISODIC, content=summary,
                                             source=MemorySource.AGENT_SUMMARY, trust_score=0.7, trace_id=trace,
                                             metadata={"thread_id": ev.get("thread_id"), "agents": agents},
                                             expires_at=datetime.now(timezone.utc) + timedelta(days=180))):
            written["episodic"] += 1
        for fact in extract_preferences(ev.get("message", "")):
            if await self.ltm.write(MemoryRecord(user_id=user, kind=MemoryKind.SEMANTIC, content=fact,
                                                 source=MemorySource.USER_STATEMENT, trust_score=0.6, trace_id=trace)):
                written["semantic"] += 1
        rejected = any(d.get("decision") == "rejected" for d in ev.get("hitl", []))
        if (ev.get("status") == "completed" and ev.get("compliance_ok", True) and not rejected
                and ev.get("plan_source") == "llm" and statuses and set(statuses.values()) <= {"completed"}):
            tools = [t for t in ev.get("trajectory", []) if ":" in t and not t.startswith(("plan", "guard", "hitl",
                                                                                            "compliance", "synth"))]
            content = (f"Request like '{mask_entities(ev.get('message', ''))[:160]}' -> agents {agents} "
                       f"(in order), tools {tools[:6]}")
            if await self.ltm.write(MemoryRecord(user_id=None, kind=MemoryKind.PROCEDURAL, content=content,
                                                 source=MemorySource.AGENT_SUMMARY, trust_score=0.7, trace_id=trace,
                                                 metadata={"prompt_version": ev.get("prompt_version")})):
                written["procedural"] += 1
        log.info("memory consolidated", thread_id=ev.get("thread_id"), **written)
        return written


async def write_audit(pool, ev: dict) -> None:
    if pool is None:
        log.info("audit", **{k: v for k, v in ev.items() if k in ("event_type", "user_id", "trace_id", "status")})
        return
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO audit_log (event_type, user_id, trace_id, payload) VALUES ($1,$2,$3,$4::jsonb)",
                           ev.get("event_type", "unknown"), ev.get("user_id"), ev.get("trace_id"),
                           json.dumps(ev, default=str))


def create_app() -> FastAPI:
    s = get_settings()
    state: dict = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        configure_logging("memory-worker", s.log_level)
        redis = None
        try:
            from redis.asyncio import Redis
            redis = Redis.from_url(s.redis_url)
            await redis.ping()
        except Exception as e:  # noqa: BLE001
            log.error("redis unavailable", error=str(e))
            redis = None
        try:
            ltm: LongTermMemory = await PgVectorLongTermMemory.create(s.postgres_dsn.get_secret_value(),
                                                                      get_embedder())
            pool = ltm.pool  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            log.error("postgres unavailable, using in-memory LTM", error=str(e))
            ltm, pool = InMemoryLongTermMemory(get_embedder()), None
        state.update(consolidator=Consolidator(ltm, redis), pool=pool)
        tasks, stop = [], asyncio.Event()
        if redis is not None and s.event_bus_backend == "redis":
            bus = RedisStreamsBus(redis)
            consumer = os.getenv("HOSTNAME", "worker-1")
            tasks.append(asyncio.create_task(bus.consume(TOPIC_MEMORY, "memory-worker", consumer,
                                                         state["consolidator"].handle, stop=stop)))
            tasks.append(asyncio.create_task(bus.consume(TOPIC_AUDIT, "audit-writer", consumer,
                                                         lambda ev: write_audit(pool, ev), stop=stop)))
        yield
        stop.set()
        for t in tasks:
            t.cancel()

    app = harden(FastAPI(title="memory-worker", lifespan=lifespan, docs_url=None), "memory-worker")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/pubsub/push")
    async def pubsub_push(request: Request):
        """Pub/Sub push subscription endpoint (OIDC-authenticated by Cloud Run IAM)."""
        envelope = await request.json()
        msg = envelope.get("message", {})
        ev = json.loads(base64.b64decode(msg.get("data", "")).decode() or "{}")
        sub = envelope.get("subscription", "")
        if "audit" in sub:
            await write_audit(state.get("pool"), ev)
            return {"ok": True}
        return await state["consolidator"].handle(ev)

    return app


app = create_app()
