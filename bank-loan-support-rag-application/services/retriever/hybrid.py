"""Hybrid search = dense (Pinecone, cosine) + sparse (BM25), fused with Reciprocal Rank Fusion.

RRF (Cormack, Clarke & Büttcher, SIGIR 2009):

    RRF(d) = Σ_{r ∈ rankers}  1 / (k + rank_r(d))          k = 60 by default

* Uses only ranks, so cosine similarities (0..1) and BM25 scores (unbounded)
  never need to be normalised against each other.
* A chunk ranked well by *both* rankers beats a chunk ranked first by only one.
* `k` dampens the influence of top ranks; 60 is the value from the paper and
  works well without tuning.
Optional per-ranker weights let you bias toward keywords or semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FusedHit:
    chunk_id: str
    rrf_score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(ranked_lists: dict[str, list[tuple[str, float]]], k: int = 60,
                           weights: dict[str, float] | None = None, top_n: int | None = None) -> list[FusedHit]:
    """`ranked_lists` maps ranker name -> [(chunk_id, raw_score), ...] sorted best-first."""
    fused: dict[str, FusedHit] = {}
    for name, hits in ranked_lists.items():
        w = (weights or {}).get(name, 1.0)
        for rank, (cid, score) in enumerate(hits, start=1):
            h = fused.setdefault(cid, FusedHit(cid))
            if name in h.ranks:            # duplicate id inside one list: keep best rank
                continue
            h.ranks[name] = rank
            h.scores[name] = score
            h.rrf_score += w / (k + rank)
    # tie-break: more rankers agreeing, then best individual rank
    out = sorted(fused.values(), key=lambda h: (-h.rrf_score, -len(h.ranks), min(h.ranks.values())))
    return out[:top_n] if top_n else out
