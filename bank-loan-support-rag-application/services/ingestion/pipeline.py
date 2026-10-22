"""Incremental ingestion pipeline: PDF -> Markdown -> chunks -> embeddings -> Pinecone."""
from __future__ import annotations

import logging
import pathlib
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import yaml

from common import storage
from common.chunk_store import load_chunks, save_chunks
from common.config import Settings
from common.embeddings import Embedder
from common.log import log_event
from common.schemas import CDCRunReport, ChangeType, Chunk, LoanProduct, utcnow
from common.vectorstore import VectorStore
from services.ingestion.cdc import CDCState, FileState, append_events, detect_file_changes, diff_chunks, sha256_bytes
from services.ingestion.chunker import StructureAwareChunker
from services.ingestion.pdf_to_markdown import pdf_bytes_to_markdown

logger = logging.getLogger(__name__)
MANIFEST = pathlib.Path(__file__).resolve().parents[2] / "data" / "sources.yaml"


def doc_id_for(source_file: str) -> str:
    stem = storage.basename(source_file).rsplit(".", 1)[0].lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "doc"


def _manifest() -> dict[str, dict]:
    if MANIFEST.exists():
        return {d["id"]: d for d in yaml.safe_load(MANIFEST.read_text())["documents"]}
    return {}


def infer_product(doc_id: str, md: str, manifest: dict[str, dict]) -> LoanProduct:
    if doc_id in manifest:
        return LoanProduct(manifest[doc_id].get("product", "general"))
    probe = (doc_id + " " + md[:3000]).lower()
    if re.search(r"home|housing", probe):
        return LoanProduct.home_loan
    if re.search(r"\bcar\b|auto loan|vehicle", probe):
        return LoanProduct.car_loan
    if "personal loan" in probe or "personal_loan" in probe:
        return LoanProduct.personal_loan
    return LoanProduct.general


class IngestionPipeline:
    def __init__(self, settings: Settings, store: VectorStore, embedder: Embedder,
                 converter: Callable[[bytes, str], str] = pdf_bytes_to_markdown) -> None:
        self.s = settings
        self.store = store
        self.embedder = embedder
        self.convert = converter
        self.chunker = StructureAwareChunker(settings.chunk_min_tokens, settings.chunk_max_tokens,
                                             settings.table_max_rows_per_chunk)
        self.manifest = _manifest()
        self.events_uri = settings.cdc_state_uri.rsplit("/", 1)[0] + "/cdc_events.jsonl"

    def run(self, force: bool = False) -> CDCRunReport:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        report = CDCRunReport(run_id=run_id)
        state = CDCState() if force else CDCState.load(self.s.cdc_state_uri)
        corpus = {c.chunk_id: c for c in ([] if force else load_chunks(self.s.chunk_store_uri))}

        files = storage.list_files(self.s.pdf_source_uri, ".pdf")
        blobs = {storage.basename(f): f for f in files}
        current = {}
        data_cache: dict[str, bytes] = {}
        for name, uri in blobs.items():
            data = storage.read_bytes(uri) or b""
            data_cache[name] = data
            current[name] = sha256_bytes(data)

        changes = detect_file_changes(state, current)
        report.file_changes = changes
        log_event(logger, "cdc file diff", run_id=run_id,
                  **{k.value: sum(c.change is k for c in changes) for k in ChangeType})

        for ch in changes:
            if ch.change is ChangeType.unchanged:
                report.chunks_unchanged += len(state.files[ch.source_file].chunk_positions)
                continue
            if ch.change is ChangeType.delete:
                old = state.files.pop(ch.source_file)
                ids = list(old.chunk_positions)
                self.store.delete(ids)
                for i in ids:
                    corpus.pop(i, None)
                report.chunks_deleted += len(ids)
                continue

            # INSERT / UPDATE
            name = ch.source_file
            doc_id = doc_id_for(name)
            md = self.convert(data_cache[name], name)
            storage.write_text(f"{self.s.markdown_out_dir}/{doc_id}.md", md)  # inspectable artefact
            title = self.manifest.get(doc_id, {}).get("title", doc_id.replace("_", " ").title())
            new_chunks = self.chunker.chunk(md, doc_id=doc_id, doc_title=title, source_file=name,
                                            product=infer_product(doc_id, md, self.manifest))
            old_positions = state.files[name].chunk_positions if name in state.files else {}
            d = diff_chunks(old_positions, new_chunks)

            if d.to_upsert:
                vectors = self.embedder.embed_documents([c.text for c in d.to_upsert])
                self.store.upsert(d.to_upsert, vectors)
            for c in d.moved:
                self.store.update_metadata(c.chunk_id, {"position": c.position, "page": c.page or 0})
            if d.to_delete:
                self.store.delete(d.to_delete)

            for i in d.to_delete:
                corpus.pop(i, None)
            for c in new_chunks:
                corpus[c.chunk_id] = c
            state.files[name] = FileState(
                file_hash=ch.new_hash or "", doc_id=doc_id,
                chunk_positions={c.chunk_id: c.position for c in new_chunks},
                ingested_at=utcnow().isoformat(),
            )
            report.chunks_upserted += len(d.to_upsert)
            report.chunks_deleted += len(d.to_delete)
            report.chunks_unchanged += d.unchanged + len(d.moved)
            log_event(logger, "document ingested", doc_id=doc_id, change=ch.change.value,
                      chunks=len(new_chunks), upserted=len(d.to_upsert), deleted=len(d.to_delete),
                      tables=sum(c.chunk_type.value == "table" for c in new_chunks))

        # Commit: chunk corpus first, then the CDC state ("log position").
        save_chunks(self.s.chunk_store_uri, list(corpus.values()))
        if report.chunks_upserted or report.chunks_deleted or any(
                c.change is not ChangeType.unchanged for c in changes):
            # tiny marker polled by every retriever instance to hot-reload BM25
            storage.write_text(self.s.chunk_store_uri + ".version", run_id)
        state.save(self.s.cdc_state_uri)
        append_events(self.events_uri, run_id, changes)
        report.finished_at = utcnow()
        log_event(logger, "ingestion run complete", **report.model_dump(mode="json", exclude={"file_changes"}))
        return report

    @staticmethod
    def corpus_stats(chunks: list[Chunk]) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in chunks:
            out[c.chunk_type.value] = out.get(c.chunk_type.value, 0) + 1
        return out
