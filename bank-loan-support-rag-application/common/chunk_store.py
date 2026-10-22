"""Persisted chunk corpus (JSONL, local or GCS).

Pinecone holds the vectors; this file holds the same chunks as text. The retriever
builds its BM25 index from it, and the ingestion CDC diff compares against it.
"""
from __future__ import annotations

import json

from common import storage
from common.schemas import Chunk


def load_chunks(uri: str) -> list[Chunk]:
    raw = storage.read_text(uri)
    if not raw:
        return []
    return [Chunk.model_validate_json(line) for line in raw.splitlines() if line.strip()]


def save_chunks(uri: str, chunks: list[Chunk]) -> None:
    ordered = sorted(chunks, key=lambda c: (c.doc_id, c.position))
    storage.write_text(uri, "\n".join(json.dumps(c.model_dump(mode="json"), ensure_ascii=False) for c in ordered) + "\n")
