"""Vector store adapters: Pinecone (serverless, free Starter plan) and in-memory (tests)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from common.config import Settings
from common.embeddings import cosine
from common.schemas import Chunk


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...
    def delete(self, ids: list[str]) -> int: ...
    def query(self, vector: list[float], top_k: int, flt: dict[str, Any] | None = None) -> list[VectorHit]: ...
    def update_metadata(self, chunk_id: str, metadata: dict[str, Any]) -> None: ...


class PineconeStore:
    BATCH = 100

    def __init__(self, settings: Settings) -> None:
        from pinecone import Pinecone, ServerlessSpec

        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required for vector_backend=pinecone")
        pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        if not pc.has_index(settings.pinecone_index):
            pc.create_index(
                name=settings.pinecone_index, dimension=settings.embedding_dim, metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        self.index = pc.Index(settings.pinecone_index)
        self.namespace = settings.pinecone_namespace

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        items = [{"id": c.chunk_id, "values": v, "metadata": c.pinecone_metadata()} for c, v in zip(chunks, vectors)]
        for i in range(0, len(items), self.BATCH):
            self.index.upsert(vectors=items[i:i + self.BATCH], namespace=self.namespace)
        return len(items)

    def delete(self, ids: list[str]) -> int:
        for i in range(0, len(ids), 1000):
            self.index.delete(ids=ids[i:i + 1000], namespace=self.namespace)
        return len(ids)

    def query(self, vector: list[float], top_k: int, flt: dict[str, Any] | None = None) -> list[VectorHit]:
        res = self.index.query(vector=vector, top_k=top_k, include_metadata=True,
                               namespace=self.namespace, filter=flt or None)
        return [VectorHit(m["id"], float(m["score"]), dict(m.get("metadata") or {})) for m in res["matches"]]

    def update_metadata(self, chunk_id: str, metadata: dict[str, Any]) -> None:
        self.index.update(id=chunk_id, set_metadata=metadata, namespace=self.namespace)


class InMemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, tuple[list[float], dict[str, Any]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        for c, v in zip(chunks, vectors):
            self._data[c.chunk_id] = (v, c.pinecone_metadata())
        return len(chunks)

    def delete(self, ids: list[str]) -> int:
        n = 0
        for i in ids:
            n += self._data.pop(i, None) is not None
        return n

    def query(self, vector: list[float], top_k: int, flt: dict[str, Any] | None = None) -> list[VectorHit]:
        def ok(md: dict[str, Any]) -> bool:
            if not flt:
                return True
            for k, cond in flt.items():
                if isinstance(cond, dict):
                    allowed = cond.get("$in", [cond.get("$eq")])
                else:
                    allowed = [cond]
                if md.get(k) not in allowed:
                    return False
            return True

        hits = [VectorHit(i, cosine(vector, v), md) for i, (v, md) in self._data.items() if ok(md)]
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]

    def update_metadata(self, chunk_id: str, metadata: dict[str, Any]) -> None:
        if chunk_id in self._data:
            self._data[chunk_id][1].update(metadata)

    def __len__(self) -> int:
        return len(self._data)


def build_vector_store(settings: Settings) -> VectorStore:
    return PineconeStore(settings) if settings.vector_backend == "pinecone" else InMemoryStore()


async def aquery(store: VectorStore, vector: list[float], top_k: int, flt: dict[str, Any] | None = None) -> list[VectorHit]:
    # The Pinecone client is sync; push it to a worker thread so the event loop never blocks.
    return await asyncio.to_thread(store.query, vector, top_k, flt)
