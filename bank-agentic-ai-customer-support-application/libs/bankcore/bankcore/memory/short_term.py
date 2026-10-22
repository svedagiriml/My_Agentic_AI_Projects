"""Short-term memory (Redis): conversation history, session state and scratch notes.

Keys (all expire after ``short_term_ttl_s`` of inactivity):
    sess:{sid}:owner   -> user id (prevents session hijacking across customers)
    sess:{sid}:history -> list of ConversationTurn JSON (bounded)
    sess:{sid}:notes   -> list of short-term notes ("customer verified via OTP", "discussing ACC-100001")
    sess:{sid}:state   -> hash of small state values (last_agent, pending_thread_id, ...)
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from redis.asyncio import Redis

from bankcore.errors import AuthorizationError
from bankcore.logging import get_logger
from bankcore.observability.metrics import MEMORY_OPS
from bankcore.schemas.memory import ConversationTurn

log = get_logger(__name__)


class ShortTermMemory:
    def __init__(self, redis: Redis, ttl_s: int = 86400, max_turns: int = 20) -> None:
        self.r, self.ttl, self.max_turns = redis, ttl_s, max_turns

    @staticmethod
    def _k(sid: str, part: str) -> str:
        return f"sess:{sid}:{part}"

    async def bind_owner(self, session_id: str, user_id: str) -> None:
        key = self._k(session_id, "owner")
        created = await self.r.set(key, user_id, nx=True, ex=self.ttl)
        if not created:
            owner = await self.r.get(key)
            owner = owner.decode() if isinstance(owner, bytes) else owner
            if owner != user_id:
                MEMORY_OPS.labels("redis", "bind_owner", "denied").inc()
                raise AuthorizationError("session belongs to another user")
            await self.r.expire(key, self.ttl)

    async def _touch(self, session_id: str) -> None:
        pipe = self.r.pipeline()
        for part in ("history", "notes", "state", "owner"):
            pipe.expire(self._k(session_id, part), self.ttl)
        await pipe.execute()

    async def append_turn(self, session_id: str, role: str, content: str, **meta: Any) -> None:
        turn = ConversationTurn(role=role, content=content[:8000], ts=time.time(), meta=meta)
        key = self._k(session_id, "history")
        pipe = self.r.pipeline()
        pipe.rpush(key, turn.model_dump_json())
        pipe.ltrim(key, -self.max_turns * 2, -1)
        await pipe.execute()
        await self._touch(session_id)
        MEMORY_OPS.labels("redis", "append_turn", "ok").inc()

    async def history(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        raw = await self.r.lrange(self._k(session_id, "history"), -(limit or self.max_turns * 2), -1)
        return [ConversationTurn.model_validate_json(x) for x in raw]

    async def add_note(self, session_id: str, note: str, max_notes: int = 15) -> None:
        key = self._k(session_id, "notes")
        pipe = self.r.pipeline()
        pipe.rpush(key, note[:300])
        pipe.ltrim(key, -max_notes, -1)
        await pipe.execute()
        await self._touch(session_id)

    async def notes(self, session_id: str) -> list[str]:
        raw = await self.r.lrange(self._k(session_id, "notes"), 0, -1)
        return [x.decode() if isinstance(x, bytes) else x for x in raw]

    async def set_state(self, session_id: str, **values: Any) -> None:
        if values:
            await self.r.hset(self._k(session_id, "state"), mapping={k: json.dumps(v) for k, v in values.items()})
            await self._touch(session_id)

    async def get_state(self, session_id: str) -> dict[str, Any]:
        raw = await self.r.hgetall(self._k(session_id, "state"))
        out = {}
        for k, v in raw.items():
            k = k.decode() if isinstance(k, bytes) else k
            out[k] = json.loads(v)
        return out

    async def clear(self, session_id: str) -> None:
        await self.r.delete(*(self._k(session_id, p) for p in ("history", "notes", "state", "owner")))


class InMemoryShortTermMemory(ShortTermMemory):
    """Process-local fallback used when Redis is unavailable (degraded mode) and in tests."""

    def __init__(self, max_turns: int = 20) -> None:  # noqa: D107 — intentionally no super()
        self.max_turns = max_turns
        self._hist: dict[str, list[ConversationTurn]] = defaultdict(list)
        self._notes: dict[str, list[str]] = defaultdict(list)
        self._state: dict[str, dict] = defaultdict(dict)
        self._owner: dict[str, str] = {}

    async def bind_owner(self, session_id: str, user_id: str) -> None:
        owner = self._owner.setdefault(session_id, user_id)
        if owner != user_id:
            raise AuthorizationError("session belongs to another user")

    async def append_turn(self, session_id: str, role: str, content: str, **meta: Any) -> None:
        self._hist[session_id].append(ConversationTurn(role=role, content=content[:8000], ts=time.time(), meta=meta))
        self._hist[session_id] = self._hist[session_id][-self.max_turns * 2:]

    async def history(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        return self._hist[session_id][-(limit or self.max_turns * 2):]

    async def add_note(self, session_id: str, note: str, max_notes: int = 15) -> None:
        self._notes[session_id] = (self._notes[session_id] + [note[:300]])[-max_notes:]

    async def notes(self, session_id: str) -> list[str]:
        return list(self._notes[session_id])

    async def set_state(self, session_id: str, **values: Any) -> None:
        self._state[session_id].update(values)

    async def get_state(self, session_id: str) -> dict[str, Any]:
        return dict(self._state[session_id])

    async def clear(self, session_id: str) -> None:
        for d in (self._hist, self._notes, self._state, self._owner):
            d.pop(session_id, None)
