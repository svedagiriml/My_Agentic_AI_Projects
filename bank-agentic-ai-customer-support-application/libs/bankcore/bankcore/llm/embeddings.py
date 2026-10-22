"""Free embedding backends: sentence-transformers (BGE-small, local), Ollama (nomic-embed-text),
and a deterministic feature-hashing embedder used for tests / degraded mode."""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx

from bankcore.config import get_settings
from bankcore.logging import get_logger

log = get_logger(__name__)


class Embedder(ABC):
    dim: int

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


class HashEmbedder(Embedder):
    """Word + char-trigram feature hashing, L2-normalised. Deterministic, dependency-free."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [padded[i:i + 3] for i in range(len(padded) - 2)]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str, dim: int) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, lazy
        self._model = SentenceTransformer(model)
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = await asyncio.to_thread(self._model.encode, texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class OllamaEmbedder(Embedder):
    def __init__(self, base_url: str, model: str, dim: int) -> None:
        self.base_url, self.model, self.dim = base_url.rstrip("/"), model, dim
        self._client = httpx.AsyncClient(timeout=15)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        r = await self._client.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    try:
        if s.embedding_backend == "sentence-transformers":
            return SentenceTransformerEmbedder(s.embedding_model, s.embedding_dim)
        if s.embedding_backend == "ollama":
            return OllamaEmbedder(s.ollama_base_url, s.embedding_model, s.embedding_dim)
    except Exception as e:  # noqa: BLE001
        log.warning("embedding backend unavailable, falling back to hash embedder", error=str(e))
    return HashEmbedder(s.embedding_dim)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
