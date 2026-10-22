from common.schemas import ChunkType, LoanProduct
from services.ingestion.chunker import StructureAwareChunker, parse_blocks
from tests.conftest import SAMPLE_MD


def _chunk(md, **kw):
    return StructureAwareChunker(min_tokens=10, **kw).chunk(md, doc_id="idbi_pl", doc_title="IDBI PL SOC",
                                                            source_file="idbi.pdf", product=LoanProduct.personal_loan)


def test_each_paragraph_and_table_is_its_own_chunk():
    chunks = _chunk(SAMPLE_MD)
    kinds = [c.chunk_type for c in chunks]
    assert kinds == [ChunkType.paragraph, ChunkType.table, ChunkType.paragraph]
    assert chunks[0].section_path[-1] == "Processing fee"
    assert chunks[1].section_path[-1] == "Pre-payment charges"


def test_table_is_linearised_for_embedding_and_markdown_for_llm():
    table = _chunk(SAMPLE_MD)[1]
    assert "Period from disbursement: Within 12 months | Charge: 5% of the amount paid" in table.text
    assert "| Within 12 months | 5% of the amount paid |" in table.display_text
    assert table.table_meta and table.table_meta["n_rows"] == 3


def test_contextual_header_prefixed():
    c = _chunk(SAMPLE_MD)[0]
    assert c.text.startswith("[IDBI PL SOC] Personal Loan Schedule of Charges > Processing fee")


def test_table_split_across_pages_is_stitched():
    md = """<!-- page: 1 -->
**Fees**

| Item | Fee |
|---|---|
| A | 1 |

<!-- page: 2 -->

| Item | Fee |
|---|---|
| B | 2 |
"""
    tables = [c for c in _chunk(md) if c.chunk_type is ChunkType.table]
    assert len(tables) == 1
    assert "| B | 2 |" in tables[0].display_text and tables[0].display_text.count("| Item | Fee |") == 1


def test_large_table_split_repeats_header():
    rows = "\n".join(f"| row{i} | {i}% |" for i in range(30))
    md = f"**Rates**\n\n| Slab | Rate |\n|---|---|\n{rows}\n"
    tables = _chunk(md, table_max_rows=10)
    assert len(tables) == 3
    assert all("| Slab | Rate |" in t.display_text for t in tables)
    assert tables[0].table_meta["total_parts"] == 3


def test_ids_are_content_addressed_and_stable():
    a, b = _chunk(SAMPLE_MD), _chunk(SAMPLE_MD)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    edited = _chunk(SAMPLE_MD.replace("8% per annum", "9% per annum"))
    assert a[0].chunk_id == edited[0].chunk_id and a[1].chunk_id == edited[1].chunk_id
    assert a[2].chunk_id != edited[2].chunk_id


def test_list_keeps_lead_in():
    md = "**Disbursement**\n\nThe loan will be disbursed only on the following conditions:\n- Title is clear\n- Margin paid\n"
    blocks = parse_blocks(md)
    assert [b.kind for b in blocks] == ["paragraph", "list"]
    chunks = _chunk(md)
    assert len(chunks) == 1 and chunks[0].chunk_type is ChunkType.list
    assert "following conditions:" in chunks[0].text and "- Margin paid" in chunks[0].text


def test_long_paragraph_split_on_sentences():
    para = " ".join(f"Sentence number {i} explains a rule about the loan." for i in range(80))
    chunks = _chunk(f"**Rules**\n\n{para}\n", max_tokens=100)
    assert len(chunks) > 1
    assert all(c.token_count <= 130 for c in chunks)


def test_repairs_placeholder_header_from_converter():
    md = "**Charges**\n\n|arges repayment:|Col2|\n|---|---|\n|Period|Charge|\n|Within 12 months|5%|\n"
    t = [c for c in _chunk(md) if c.chunk_type is ChunkType.table][0]
    assert t.table_meta["columns"] == ["Period", "Charge"]
    assert "Period: Within 12 months | Charge: 5%" in t.text
