"""Free, local embeddings.

Default: BAAI/bge-small-en-v1.5 via sentence-transformers (MIT, 384-d, runs on CPU,
top of its size class on MTEB retrieval). Vectors are L2-normalised so Pinecone's
cosine metric == dot product. BGE wants an instruction prefix on *queries only*.

`HashingEmbedder` is a dependency-free fallback used in unit tests / CI.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Protocol

from common.config import Settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: int, query_prefix: str = "") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = dim
        self.query_prefix = query_prefix
        actual = self.model.get_sentence_embedding_dimension()
        if actual != dim:
            raise ValueError(f"EMBEDDING_DIM={dim} but {model_name} produces {actual}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0].tolist()


class HashingEmbedder:
    """Feature-hashing bag-of-words embedder (deterministic, no downloads)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def build_embedder(settings: Settings, testing: bool = False) -> Embedder:
    if testing:
        return HashingEmbedder(settings.embedding_dim)
    return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim,
                                       settings.embedding_query_prefix)


async def aembed_query(embedder: Embedder, text: str) -> list[float]:
    # Model inference is CPU-bound: run it off the event loop so the worker stays responsive.
    return await asyncio.to_thread(embedder.embed_query, text)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
