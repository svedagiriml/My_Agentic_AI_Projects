"""Memory record contracts (long-term memory in Postgres + PGVector)."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import Field

from bankcore.schemas.common import StrictModel


class MemoryKind(str, Enum):
    EPISODIC = "episodic"      # what happened: summaries of past conversations / outcomes
    SEMANTIC = "semantic"      # what is true: durable facts & preferences about the customer
    PROCEDURAL = "procedural"  # how to do it: successful plans / tool trajectories per intent


class MemorySource(str, Enum):
    USER_STATEMENT = "user_statement"
    TOOL_VERIFIED = "tool_verified"     # came from a system of record (highest trust)
    AGENT_SUMMARY = "agent_summary"
    STAFF = "staff"


TRUST_BY_SOURCE = {
    MemorySource.TOOL_VERIFIED: 0.95,
    MemorySource.STAFF: 0.9,
    MemorySource.AGENT_SUMMARY: 0.7,
    MemorySource.USER_STATEMENT: 0.6,
}


class MemoryRecord(StrictModel):
    id: str | None = None
    user_id: str | None            # None => global (procedural memory is shared)
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: MemorySource
    trust_score: float = Field(ge=0, le=1)
    trace_id: str | None = None
    content_sha256: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None


class RetrievedMemory(StrictModel):
    kind: MemoryKind
    content: str
    score: float
    trust_score: float
    created_at: datetime


class ConversationTurn(StrictModel):
    role: str = Field(pattern=r"^(user|assistant|system)$")
    content: str = Field(max_length=8000)
    ts: float
    meta: dict[str, Any] = Field(default_factory=dict)
