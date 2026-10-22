"""Free embedding backends.

* sentence_transformers : all-MiniLM-L6-v2 (384-d, Apache-2.0) – production default
* hash                  : deterministic feature-hashing of word unigrams/bigrams +
                          char trigrams. Zero dependencies; used for tests and as a
                          fallback when the model can't be loaded.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

from loanflow.common.config import get_settings

_TOKEN = re.compile(r"[a-z0-9_]+")


def _hash_embed(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    t = text.lower()
    toks = _TOKEN.findall(t)
    feats: list[tuple[str, float]] = [(w, 1.0) for w in toks]
    feats += [(f"{a}|{b}", 0.7) for a, b in zip(toks, toks[1:])]
    for w in toks:
        padded = f"#{w}#"
        feats += [(padded[i : i + 3], 0.3) for i in range(len(padded) - 2)]
    for f, weight in feats:
        h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
        vec[h % dim] += weight if (h >> 63) & 1 else -weight
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@lru_cache(maxsize=1)
def _st_model(name: str):
    from sentence_transformers import SentenceTransformer  # lazy, heavy

    return SentenceTransformer(name)


def embed(text: str) -> list[float]:
    s = get_settings()
    if s.embedding_backend == "sentence_transformers":
        try:
            return [float(x) for x in _st_model(s.embedding_model).encode(text, normalize_embeddings=True)]
        except Exception:  # noqa: BLE001 - degrade gracefully
            pass
    return _hash_embed(text, s.embedding_dim)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
