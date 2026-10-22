from common.embeddings import HashingEmbedder
from common.schemas import LoanProduct, RetrieveRequest
from common.vectorstore import InMemoryStore
from services.ingestion.chunker import StructureAwareChunker
from services.retriever.bm25_index import BM25Index, tokenize
from services.retriever.hybrid import reciprocal_rank_fusion
from services.retriever.service import HybridRetriever
from tests.conftest import SAMPLE_MD


def test_rrf_rewards_agreement_and_ignores_score_scales():
    fused = reciprocal_rank_fusion({"dense": [("a", 0.91), ("b", 0.90), ("c", 0.2)],
                                    "bm25": [("b", 42.0), ("d", 30.0), ("a", 1.0)]}, k=60)
    assert fused[0].chunk_id == "b"          # rank 2 + rank 1 beats rank 1 + rank 3
    assert abs(fused[0].rrf_score - (1 / 62 + 1 / 61)) < 1e-12
    assert fused[0].ranks == {"dense": 2, "bm25": 1}
    assert {h.chunk_id for h in fused} == {"a", "b", "c", "d"}


def test_rrf_weights():
    fused = reciprocal_rank_fusion({"dense": [("a", 1)], "bm25": [("b", 1)]}, weights={"bm25": 2.0})
    assert fused[0].chunk_id == "b"


def test_tokenizer_keeps_banking_tokens():
    toks = tokenize("Processing fee 0.35% for loans up to Rs.30,00,000 (EBLR linked)")
    assert "0.35%" in toks and "inr" in toks and "3000000" in toks and "eblr" in toks


def _chunks():
    return StructureAwareChunker(min_tokens=10).chunk(SAMPLE_MD, doc_id="idbi", doc_title="IDBI",
                                                      source_file="idbi.pdf", product=LoanProduct.personal_loan)


def test_bm25_finds_exact_term():
    idx = BM25Index(_chunks())
    top = idx.search("penal charges overdue", 3)
    assert "Penal" in idx.get(top[0][0]).text


async def test_hybrid_retriever_end_to_end(settings):
    chunks = _chunks()
    emb, store = HashingEmbedder(), InMemoryStore()
    store.upsert(chunks, emb.embed_documents([c.text for c in chunks]))
    r = HybridRetriever(settings, store, emb)
    r.reload(chunks)
    resp = await r.retrieve(RetrieveRequest(query="prepayment charge within 12 months", top_k=3,
                                            product_filter=LoanProduct.personal_loan))
    assert resp.chunks[0].chunk_type.value == "table"
    top = resp.chunks[0]
    assert top.final_rank == 1 and top.rrf_score > 0 and (top.dense_rank or top.bm25_rank)

    # product filter excludes other products (general docs would still be allowed)
    resp2 = await r.retrieve(RetrieveRequest(query="prepayment", product_filter=LoanProduct.car_loan))
    assert resp2.chunks == []
