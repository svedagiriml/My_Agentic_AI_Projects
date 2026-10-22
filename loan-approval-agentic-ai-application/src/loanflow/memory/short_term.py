"""Short-term (working) memory — Redis.

Scope: one loan run. Holds
  * run status snapshot (what the API serves on GET /runs/{id})
  * conversation history: every supervisor<->agent message (JSON), capped
  * short-term notes: scratchpad facts the supervisor wants later nodes to see
Everything here expires after SHORT_TERM_TTL_S. Durable state lives in the
LangGraph checkpointer (Postgres) and long-term memory (PGVector).
"""
from __future__ import annotations

from typing import Any

from loanflow.common.config import get_settings
from loanflow.memory.kv import get_kv
from loanflow.security.guardrails import redact_pii

HISTORY_MAX = 200


class ShortTermMemory:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.kv = get_kv()
        self.ttl = get_settings().short_term_ttl_s

    def _k(self, part: str) -> str:
        return f"stm:{self.run_id}:{part}"

    async def append_message(self, role: str, content: dict[str, Any]) -> None:
        await self.kv.rpush(self._k("history"), {"role": role, "content": content}, ttl_s=self.ttl, maxlen=HISTORY_MAX)

    async def history(self, last_n: int = 20) -> list[dict[str, Any]]:
        return await self.kv.lrange(self._k("history"), -last_n, -1)

    async def add_note(self, note: str) -> None:
        await self.kv.rpush(self._k("notes"), redact_pii(note).text[:500], ttl_s=self.ttl, maxlen=50)

    async def notes(self) -> list[str]:
        return await self.kv.lrange(self._k("notes"))

    async def save_status(self, status: dict[str, Any]) -> None:
        await self.kv.set(self._k("status"), status, ttl_s=self.ttl)

    async def status(self) -> dict[str, Any] | None:
        return await self.kv.get(self._k("status"))


async def find_run_by_idempotency_key(key: str) -> str | None:
    return await get_kv().get(f"idem:{key}")


async def remember_idempotency_key(key: str, run_id: str) -> bool:
    """Returns False if the key already maps to another run."""
    kv = get_kv()
    existing = await kv.get(f"idem:{key}")
    if existing:
        return existing == run_id
    await kv.set(f"idem:{key}", run_id, ttl_s=86_400)
    return True
