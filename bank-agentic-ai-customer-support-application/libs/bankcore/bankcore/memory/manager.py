"""MemoryManager — single facade the supervisor uses to read/write all memory tiers.

Reads degrade gracefully: if Redis or Postgres is down the conversation continues with
empty memory (logged + metered) rather than failing the customer's request.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bankcore.logging import get_logger
from bankcore.memory.long_term import LongTermMemory
from bankcore.memory.short_term import ShortTermMemory
from bankcore.observability.metrics import MEMORY_OPS
from bankcore.schemas.memory import ConversationTurn, MemoryKind, RetrievedMemory

log = get_logger(__name__)


@dataclass
class MemoryContext:
    history: list[ConversationTurn] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    episodic: list[RetrievedMemory] = field(default_factory=list)
    semantic: list[RetrievedMemory] = field(default_factory=list)
    procedural: list[RetrievedMemory] = field(default_factory=list)


class MemoryManager:
    def __init__(self, short_term: ShortTermMemory, long_term: LongTermMemory | None) -> None:
        self.st, self.lt = short_term, long_term

    async def load(self, *, user_id: str, session_id: str, query: str) -> MemoryContext:
        ctx = MemoryContext()
        try:
            ctx.history = await self.st.history(session_id, limit=12)
            ctx.notes = await self.st.notes(session_id)
            ctx.state = await self.st.get_state(session_id)
        except Exception as e:  # noqa: BLE001
            MEMORY_OPS.labels("redis", "load", "error").inc()
            log.warning("short-term memory unavailable", error=str(e))
        if self.lt is not None:
            try:
                found = await self.lt.search(user_id=user_id, query=query, k=6,
                                             kinds=[MemoryKind.EPISODIC, MemoryKind.SEMANTIC, MemoryKind.PROCEDURAL])
                ctx.episodic = [m for m in found if m.kind == MemoryKind.EPISODIC][:3]
                ctx.semantic = [m for m in found if m.kind == MemoryKind.SEMANTIC][:4]
                ctx.procedural = [m for m in found if m.kind == MemoryKind.PROCEDURAL][:2]
            except Exception as e:  # noqa: BLE001
                MEMORY_OPS.labels("pgvector", "load", "error").inc()
                log.warning("long-term memory unavailable", error=str(e))
        return ctx

    async def record_turn(self, session_id: str, user_text: str, assistant_text: str, **meta) -> None:
        try:
            await self.st.append_turn(session_id, "user", user_text)
            await self.st.append_turn(session_id, "assistant", assistant_text, **meta)
        except Exception as e:  # noqa: BLE001
            log.warning("failed to persist short-term turn", error=str(e))
