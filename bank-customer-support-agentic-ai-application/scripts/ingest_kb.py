"""Ingest data/kb/*.md into Postgres ``kb_chunks`` (pgvector + tsvector) for the knowledge MCP server.

    python scripts/ingest_kb.py --dsn postgresql://bank:bank@localhost:5432/bank
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services"))

from bankcore.llm.embeddings import get_embedder  # noqa: E402

from mcp_servers.knowledge.kb_index import InMemoryKnowledgeIndex  # noqa: E402


async def main(dsn: str, kb_dir: Path) -> None:
    import asyncpg
    import numpy as np
    from pgvector.asyncpg import register_vector

    idx = InMemoryKnowledgeIndex(get_embedder())
    n = await idx.ingest_dir(kb_dir)
    conn = await asyncpg.connect(dsn)
    await register_vector(conn)
    async with conn.transaction():
        for chunk, vec in zip(idx.chunks, idx.vectors):
            await conn.execute(
                """INSERT INTO kb_chunks (chunk_id, doc_id, title, content, source, embedding)
                   VALUES ($1,$2,$3,$4,$5,$6)
                   ON CONFLICT (chunk_id) DO UPDATE SET content=EXCLUDED.content, title=EXCLUDED.title,
                       embedding=EXCLUDED.embedding, updated_at=now()""",
                chunk.doc_id, chunk.doc_id.split("#")[0], chunk.title, chunk.content, chunk.source,
                np.array(vec, dtype=np.float32))
    await conn.close()
    print(f"ingested {n} chunks; quarantined {len(idx.quarantined)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="postgresql://bank:bank@localhost:5432/bank")
    ap.add_argument("--kb-dir", default=str(ROOT / "data" / "kb"))
    a = ap.parse_args()
    asyncio.run(main(a.dsn, Path(a.kb_dir)))
