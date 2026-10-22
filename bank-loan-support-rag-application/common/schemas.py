"""Pydantic contracts shared by all services.

Strict models (`extra="forbid"`, constrained types) are used for every request,
response and LLM structured output. Anything that crosses a service boundary or
comes back from an LLM is validated here, so malformed data fails loudly at the
edge instead of propagating.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


def utcnow() -> datetime:
    return datetime.now(UTC)


QueryText = Annotated[str, StringConstraints(min_length=2, max_length=2000)]


class LoanProduct(str, Enum):
    home_loan = "home_loan"
    personal_loan = "personal_loan"
    car_loan = "car_loan"
    general = "general"


class ChunkType(str, Enum):
    paragraph = "paragraph"
    table = "table"
    list = "list"


# --------------------------------------------------------------------------- chunks
class Chunk(StrictModel):
    chunk_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+::[0-9a-f]{16}$")]
    doc_id: str
    source_file: str
    product: LoanProduct = LoanProduct.general
    chunk_type: ChunkType
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None
    position: int = Field(ge=0, description="order of the chunk inside the document")
    text: str = Field(min_length=1, description="text that is embedded / BM25-indexed")
    display_text: str = Field(min_length=1, description="original markdown shown to the LLM")
    content_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    token_count: int = Field(ge=1)
    table_meta: dict[str, object] | None = None

    def pinecone_metadata(self) -> dict[str, object]:
        # Pinecone metadata must be flat (str/num/bool/list[str]) and < 40 KB.
        md: dict[str, object] = {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "product": self.product.value,
            "chunk_type": self.chunk_type.value,
            "section_path": self.section_path,
            "position": self.position,
            "content_hash": self.content_hash,
            "display_text": self.display_text[:30000],
        }
        if self.page is not None:
            md["page"] = self.page
        return md


# --------------------------------------------------------------------------- guardrail
class PIIEntity(StrictModel):
    entity_type: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    placeholder: str


class GuardrailRequest(StrictModel):
    text: Annotated[str, StringConstraints(max_length=20000)]
    direction: Literal["input", "output"]


class GuardrailResponse(StrictModel):
    sanitized_text: str
    pii_found: list[PIIEntity] = Field(default_factory=list)
    injection_detected: bool = False
    injection_score: float = Field(0.0, ge=0.0, le=1.0)
    injection_reasons: list[str] = Field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None


# --------------------------------------------------------------------------- retrieval
class RetrieveRequest(StrictModel):
    query: QueryText
    product_filter: LoanProduct | None = None
    top_k: int = Field(5, ge=1, le=20)
    trace_id: str | None = None


class RetrievedChunk(StrictModel):
    chunk_id: str
    doc_id: str
    source_file: str
    chunk_type: ChunkType
    section_path: list[str]
    page: int | None = None
    display_text: str
    dense_score: float | None = Field(None, description="cosine similarity from Pinecone")
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float = Field(ge=0.0)
    final_rank: int = Field(ge=1)


class RetrieveResponse(StrictModel):
    query: str
    chunks: list[RetrievedChunk]
    latency_ms: float
    dense_candidates: int
    bm25_candidates: int


class EmbedRequest(StrictModel):
    texts: list[Annotated[str, StringConstraints(min_length=1, max_length=4000)]] = Field(min_length=1, max_length=64)
    is_query: bool = True


class EmbedResponse(StrictModel):
    vectors: list[list[float]]
    dim: int


# --------------------------------------------------------------------------- generation
class ChatTurn(StrictModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(max_length=4000)]


class RewriteRequest(StrictModel):
    query: QueryText
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)
    trace_id: str | None = None


class RewrittenQuery(StrictModel):
    """Structured output the LLM must return for query rewriting."""

    rewritten_query: Annotated[str, StringConstraints(min_length=2, max_length=500)] = Field(
        description="Standalone, keyword-rich search query that resolves pronouns from history"
    )
    product: LoanProduct = Field(description="Loan product the question is about; 'general' if unclear")
    intent: Literal["eligibility", "interest_rate", "fees_charges", "documents", "repayment",
                    "prepayment_foreclosure", "process", "regulation", "other"]
    is_banking_question: bool = Field(description="False if the question is unrelated to loans/banking")


class GenerateRequest(StrictModel):
    query: QueryText
    rewritten_query: str | None = None
    chunks: list[RetrievedChunk] = Field(max_length=20)
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)
    stream: bool = False
    trace_id: str | None = None


class Citation(StrictModel):
    chunk_id: str
    source_file: str
    page: int | None = None


class GeneratedAnswer(StrictModel):
    """Structured output parsed from the LLM for non-streaming generation."""

    answer: Annotated[str, StringConstraints(min_length=1, max_length=6000)]
    cited_sources: list[Annotated[str, StringConstraints(pattern=r"^S\d{1,2}$")]] = Field(
        default_factory=list, description="Labels of the context passages used, e.g. ['S1','S3']"
    )
    confidence: Literal["high", "medium", "low"]
    answerable_from_context: bool
    needs_human_agent: bool = Field(False, description="True if customer should be routed to a human")

    @field_validator("cited_sources")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


class TokenUsage(StrictModel):
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class GenerateResponse(StrictModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    answerable_from_context: bool
    needs_human_agent: bool
    model: str
    usage: TokenUsage
    cost_usd: float = Field(ge=0)
    latency_ms: float
    prompt_preview: str | None = None


# --------------------------------------------------------------------------- gateway
class ChatRequest(StrictModel):
    query: QueryText
    session_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")] = "anonymous"
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)
    stream: bool = True


class ChatResponse(StrictModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    cached: bool
    trace_id: str
    latency_ms: float
    blocked: bool = False


class SSEEvent(StrictModel):
    event: Literal["meta", "token", "citations", "done", "error"]
    data: dict[str, object] | str

    def encode(self) -> str:
        import json

        payload = self.data if isinstance(self.data, str) else json.dumps(self.data, default=str)
        lines = "\n".join(f"data: {line}" for line in payload.split("\n"))
        return f"event: {self.event}\n{lines}\n\n"


# --------------------------------------------------------------------------- CDC
class ChangeType(str, Enum):
    insert = "insert"
    update = "update"
    delete = "delete"
    unchanged = "unchanged"


class FileChange(StrictModel):
    source_file: str
    change: ChangeType
    old_hash: str | None = None
    new_hash: str | None = None


class CDCRunReport(StrictModel):
    run_id: str
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    file_changes: list[FileChange] = Field(default_factory=list)
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    chunks_unchanged: int = 0

    @model_validator(mode="after")
    def _non_negative(self) -> CDCRunReport:
        if min(self.chunks_upserted, self.chunks_deleted, self.chunks_unchanged) < 0:
            raise ValueError("counts must be >= 0")
        return self


# The prompt labels context passages [S1], [S2] ... (short labels are far more
# reliable for small LLMs than 30-char chunk IDs); we map them back to chunk IDs.
CITATION_RE = re.compile(r"\[S(\d{1,2})\]")
