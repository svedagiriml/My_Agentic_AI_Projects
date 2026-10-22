"""In-process BM25 (Okapi) keyword index over the chunk corpus.

Why BM25 next to dense search? Banking questions are full of exact tokens that
embeddings blur: "0.35%", "Rs.30 lakh", "MCLR", "EBLR", "NACH", "CERSAI", "KFS".
BM25 rewards exact term overlap (with IDF weighting and length normalisation), so
it reliably surfaces the fee-table row that contains the literal term.

The corpus (a few thousand chunks) fits easily in memory; it is rebuilt from the
chunk store on start-up and on `/admin/reload` after an ingestion run.
"""
from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

from common.schemas import Chunk

STOPWORDS = frozenset("""a an and are as at be by can do does for from has have how i if in is it its
me my of on or our shall should that the their them then there these this to was what when where which
who will with you your would may also any such other than""".split())
TOKEN_RE = re.compile(r"₹|rs\.?|\d+(?:\.\d+)?%?|[a-z][a-z0-9]*")


def _stem(tok: str) -> str:
    for suf in ("ings", "ing", "ies", "es", "s", "ed"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall(text.lower().replace(",", ""))
    out = []
    for t in toks:
        if t in STOPWORDS:
            continue
        if t in ("rs", "rs.", "₹"):
            t = "inr"
        out.append(_stem(t))
    return out


class BM25Index:
    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        if chunks:
            self.build(chunks)

    def build(self, chunks: list[Chunk]) -> None:
        corpus = [tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(corpus, k1=1.5, b=0.75) if chunks else None
        with self._lock:                       # atomic swap: queries never see a half-built index
            self._chunks, self._bm25 = list(chunks), bm25

    def __len__(self) -> int:
        return len(self._chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            return next((c for c in self._chunks if c.chunk_id == chunk_id), None)

    @property
    def by_id(self) -> dict[str, Chunk]:
        with self._lock:
            return {c.chunk_id: c for c in self._chunks}

    def search(self, query: str, top_k: int, products: set[str] | None = None) -> list[tuple[str, float]]:
        with self._lock:
            bm25, chunks = self._bm25, self._chunks
        if bm25 is None:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = bm25.get_scores(q)
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        out: list[tuple[str, float]] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if products and chunks[i].product.value not in products:
                continue
            out.append((chunks[i].chunk_id, float(scores[i])))
            if len(out) >= top_k:
                break
        return out
