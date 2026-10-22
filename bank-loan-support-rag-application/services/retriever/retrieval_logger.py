"""Per-query retrieval log.

Every query writes one JSON record with the (rewritten) query and, for each
returned chunk: id, source, page, dense score + rank, BM25 score + rank, RRF
score and final rank. Records go to:
  * stdout as structured JSON  -> Cloud Logging (filter `jsonPayload.event="retrieval"`,
    or route to BigQuery with a log sink for offline analysis)
  * a local JSONL file (RETRIEVAL_LOG_PATH) for development / RAGAS debugging
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from datetime import UTC, datetime

from common.log import log_event
from common.schemas import RetrieveResponse

logger = logging.getLogger("retrieval")


class RetrievalLogger:
    def __init__(self, path: str | None) -> None:
        self.path = pathlib.Path(path) if path and not path.startswith("gs://") else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def log(self, resp: RetrieveResponse, *, trace_id: str | None, product_filter: str | None) -> None:
        record = {
            "event": "retrieval",
            "ts": datetime.now(UTC).isoformat(),
            "trace_id": trace_id,
            "query": resp.query,
            "product_filter": product_filter,
            "latency_ms": resp.latency_ms,
            "dense_candidates": resp.dense_candidates,
            "bm25_candidates": resp.bm25_candidates,
            "chunks": [
                {"rank": c.final_rank, "chunk_id": c.chunk_id, "source_file": c.source_file, "page": c.page,
                 "type": c.chunk_type.value, "rrf": round(c.rrf_score, 6),
                 "dense_score": c.dense_score, "dense_rank": c.dense_rank,
                 "bm25_score": c.bm25_score, "bm25_rank": c.bm25_rank}
                for c in resp.chunks
            ],
        }
        log_event(logger, "retrieval", **record)
        if self.path:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            async with self._lock:
                await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        assert self.path is not None
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
