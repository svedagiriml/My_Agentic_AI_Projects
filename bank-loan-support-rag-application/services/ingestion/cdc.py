"""Change Data Capture for the PDF knowledge base.

Two levels of change detection make ingestion incremental:

1. **File level** – a SHA-256 of every PDF is compared with the last committed
   state. Unchanged files are skipped entirely (no conversion, no embedding).
   New -> INSERT, different hash -> UPDATE, missing -> DELETE.

2. **Chunk level** (for INSERT/UPDATE files) – chunk IDs are content-addressed
   (`doc_id::sha256(content)[:16]`), so diffing the new chunk-ID set against the
   old one yields exactly which chunks to upsert (new/edited paragraphs or
   tables), which to delete (removed content) and which to leave alone. Only
   the changed chunks are embedded – editing one row of a fee table re-embeds
   one chunk, not the whole document.

The state file (local JSON or GCS object) is the CDC "log position": it is only
written after the vector store has been updated, so a crash mid-run simply
re-processes the same changes next time (upserts/deletes are idempotent).
Every run also appends its change events to `cdc_events.jsonl` (audit trail).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from common import storage
from common.schemas import ChangeType, Chunk, FileChange


@dataclass
class FileState:
    file_hash: str
    doc_id: str
    chunk_positions: dict[str, int]          # chunk_id -> position
    ingested_at: str


@dataclass
class CDCState:
    files: dict[str, FileState] = field(default_factory=dict)

    @classmethod
    def load(cls, uri: str) -> CDCState:
        raw = storage.read_text(uri)
        if not raw:
            return cls()
        data = json.loads(raw)
        return cls(files={k: FileState(**v) for k, v in data.get("files", {}).items()})

    def save(self, uri: str) -> None:
        payload = {"version": 1, "updated_at": datetime.now(UTC).isoformat(),
                   "files": {k: vars(v) for k, v in sorted(self.files.items())}}
        storage.write_text(uri, json.dumps(payload, indent=2))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_file_changes(state: CDCState, current: dict[str, str]) -> list[FileChange]:
    """`current` maps source_file -> sha256 of the file bytes."""
    changes: list[FileChange] = []
    for name, h in sorted(current.items()):
        old = state.files.get(name)
        if old is None:
            changes.append(FileChange(source_file=name, change=ChangeType.insert, new_hash=h))
        elif old.file_hash != h:
            changes.append(FileChange(source_file=name, change=ChangeType.update, old_hash=old.file_hash, new_hash=h))
        else:
            changes.append(FileChange(source_file=name, change=ChangeType.unchanged, old_hash=h, new_hash=h))
    for name, old in sorted(state.files.items()):
        if name not in current:
            changes.append(FileChange(source_file=name, change=ChangeType.delete, old_hash=old.file_hash))
    return changes


@dataclass
class ChunkDiff:
    to_upsert: list[Chunk]
    to_delete: list[str]
    moved: list[Chunk]          # same content, new position -> metadata-only update
    unchanged: int


def diff_chunks(old_positions: dict[str, int], new_chunks: list[Chunk]) -> ChunkDiff:
    new_ids = {c.chunk_id for c in new_chunks}
    to_upsert = [c for c in new_chunks if c.chunk_id not in old_positions]
    moved = [c for c in new_chunks if c.chunk_id in old_positions and old_positions[c.chunk_id] != c.position]
    unchanged = sum(1 for c in new_chunks if old_positions.get(c.chunk_id) == c.position)
    to_delete = sorted(set(old_positions) - new_ids)
    return ChunkDiff(to_upsert, to_delete, moved, unchanged)


def append_events(uri: str, run_id: str, changes: list[FileChange]) -> None:
    existing = storage.read_text(uri) or ""
    ts = datetime.now(UTC).isoformat()
    lines = [json.dumps({"run_id": run_id, "ts": ts, **c.model_dump(mode="json")})
             for c in changes if c.change is not ChangeType.unchanged]
    if lines:
        storage.write_text(uri, existing + "\n".join(lines) + "\n")
