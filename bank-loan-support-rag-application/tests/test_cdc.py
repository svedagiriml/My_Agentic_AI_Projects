import pathlib

from common.chunk_store import load_chunks
from common.embeddings import HashingEmbedder
from common.schemas import ChangeType
from common.vectorstore import InMemoryStore
from services.ingestion.pipeline import IngestionPipeline
from tests.conftest import SAMPLE_MD


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__(384)
        self.embedded = 0

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return super().embed_documents(texts)


def _md_converter(data: bytes, _name: str) -> str:   # "PDF" files in tests hold markdown
    return data.decode()


def _pipeline(settings):
    store, emb = InMemoryStore(), CountingEmbedder()
    return IngestionPipeline(settings, store, emb, converter=_md_converter), store, emb


def test_incremental_cdc_lifecycle(settings):
    pdfs = pathlib.Path(settings.pdf_source_uri)
    pdfs.mkdir(parents=True)
    (pdfs / "idbi_personal_loan_soc.pdf").write_text(SAMPLE_MD)
    (pdfs / "other.pdf").write_text("# Car Loan\n\nThe car loan is disbursed to the dealer by RTGS or NEFT transfer only.\n")
    pipe, store, emb = _pipeline(settings)

    # 1) initial load: everything inserted
    r1 = pipe.run()
    assert {c.change for c in r1.file_changes} == {ChangeType.insert}
    n = len(store)
    assert r1.chunks_upserted == n == emb.embedded

    # 2) nothing changed: zero embeddings, zero writes
    r2 = pipe.run()
    assert {c.change for c in r2.file_changes} == {ChangeType.unchanged}
    assert r2.chunks_upserted == 0 and emb.embedded == n

    # 3) edit one paragraph: exactly one chunk re-embedded, one stale chunk deleted
    (pdfs / "idbi_personal_loan_soc.pdf").write_text(SAMPLE_MD.replace("8% per annum", "9% per annum"))
    r3 = pipe.run()
    upd = [c for c in r3.file_changes if c.change is ChangeType.update]
    assert len(upd) == 1
    assert r3.chunks_upserted == 1 and r3.chunks_deleted == 1
    assert emb.embedded == n + 1 and len(store) == n
    assert any("9% per annum" in c.text for c in load_chunks(settings.chunk_store_uri))

    # 4) delete a file: its chunks vanish from vector store and chunk store
    (pdfs / "other.pdf").unlink()
    r4 = pipe.run()
    assert any(c.change is ChangeType.delete for c in r4.file_changes)
    assert all(c.doc_id != "other" for c in load_chunks(settings.chunk_store_uri))
    assert len(store) == len(load_chunks(settings.chunk_store_uri))
