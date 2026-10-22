import os

import pytest

os.environ.setdefault("VECTOR_BACKEND", "memory")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")

from common.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        vector_backend="memory",
        chunk_store_uri=str(tmp_path / "state" / "chunks.jsonl"),
        cdc_state_uri=str(tmp_path / "state" / "cdc_state.json"),
        pdf_source_uri=str(tmp_path / "pdfs"),
        markdown_out_dir=str(tmp_path / "md"),
        retrieval_log_path=str(tmp_path / "logs" / "retrievals.jsonl"),
        chunk_min_tokens=10,
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )


SAMPLE_MD = """<!-- page: 1 -->
# Personal Loan Schedule of Charges

**Processing fee**

The bank charges a processing fee of 1.00% of the loan amount subject to a minimum of Rs.2500 plus applicable taxes.

**Pre-payment charges**

| Period from disbursement | Charge |
|---|---|
| Within 12 months | 5% of the amount paid |
| 12 to 24 months | 3% of the amount paid |
| After 24 months | Nil |

**Penal charges**

Penal charges of 8% per annum are levied on the overdue amount for the overdue period.
"""
