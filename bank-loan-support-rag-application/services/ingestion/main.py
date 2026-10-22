"""Ingestion entrypoints.

CLI (local, or as a Cloud Run *Job*):
    python -m services.ingestion.main            # incremental (CDC)
    python -m services.ingestion.main --force    # full rebuild

HTTP (Cloud Run *service*, target of an Eventarc trigger on the GCS bucket):
    uvicorn services.ingestion.main:app --port 8004
    POST /events/gcs   <- CloudEvent "google.cloud.storage.object.v1.finalized|deleted"
    POST /ingest       <- manual trigger
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request

from common.app_factory import create_app
from common.config import get_settings
from common.embeddings import build_embedder
from common.log import log_event, setup_logging
from common.vectorstore import build_vector_store
from services.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger("ingestion")
settings = get_settings()
_lock = asyncio.Lock()
_pipeline: IngestionPipeline | None = None


def _get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(settings, build_vector_store(settings), build_embedder(settings))
    return _pipeline


async def _notify_retriever() -> None:
    """Ask the retriever to reload its BM25 corpus (best effort)."""
    try:
        from common.http_client import ServiceClient

        client = ServiceClient("retriever", str(settings.retriever_url), use_id_token=settings.use_gcp_id_token)
        await client.get_json("/admin/reload")
        await client.aclose()
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "retriever reload failed (it will pick changes up on next restart)",
                  logging.WARNING, error=str(exc))


async def _run(force: bool = False) -> dict:
    if _lock.locked():
        return {"status": "already_running"}
    async with _lock:  # a single CDC run at a time per instance (run Cloud Run with max-instances=1)
        report = await asyncio.to_thread(_get_pipeline().run, force)
    await _notify_retriever()
    return json.loads(report.model_dump_json())


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    yield


app = create_app("ingestion", settings, lifespan)


@app.post("/ingest")
async def ingest(force: bool = False) -> dict:
    return await _run(force)


@app.post("/events/gcs")
async def gcs_event(request: Request, background: BackgroundTasks) -> dict:
    # Eventarc delivers CloudEvents in binary mode: ce-* headers + JSON body (GCS object).
    body = await request.json()
    name = body.get("name", "")
    log_event(logger, "gcs event", ce_type=request.headers.get("ce-type"), object=name)
    if not name.lower().endswith(".pdf"):
        return {"status": "ignored"}
    background.add_task(_run, False)   # ack fast; CDC figures out what changed
    return {"status": "scheduled"}


def cli() -> None:
    setup_logging("ingestion", settings.log_level)
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    report = _get_pipeline().run(force=args.force)
    print(report.model_dump_json(indent=2))
    if report.chunks_upserted or report.chunks_deleted:
        asyncio.run(_notify_retriever())


if __name__ == "__main__":
    cli()
