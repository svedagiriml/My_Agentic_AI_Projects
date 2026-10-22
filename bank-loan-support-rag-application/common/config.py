"""Typed, validated configuration (pydantic-settings).

Every service reads env vars (or a local .env). On Cloud Run, secrets are mounted
from Secret Manager as env vars, so the same code works locally and in GCP.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- general ---------------------------------------------------------
    environment: Literal["local", "dev", "prod"] = "local"
    service_name: str = "bank-rag"
    log_level: str = "INFO"

    # ---- embeddings (free, local) ----------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # ---- vector DB -------------------------------------------------------
    vector_backend: Literal["pinecone", "memory"] = "pinecone"
    pinecone_api_key: SecretStr | None = None
    pinecone_index: str = "bank-loan-support"
    pinecone_namespace: str = "loans"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ---- chunk store (feeds BM25 + CDC) -----------------------------------
    chunk_store_uri: str = "data/state/chunks.jsonl"   # or gs://bucket/state/chunks.jsonl
    cdc_state_uri: str = "data/state/cdc_state.json"   # or gs://bucket/state/cdc_state.json
    pdf_source_uri: str = "data/pdfs"                  # or gs://bucket/pdfs
    markdown_out_dir: str = "data/markdown"

    # ---- chunking --------------------------------------------------------
    chunk_min_tokens: int = Field(40, ge=5)
    chunk_max_tokens: int = Field(350, ge=50)
    table_max_rows_per_chunk: int = Field(15, ge=2)

    # ---- retrieval -------------------------------------------------------
    dense_top_k: int = Field(20, ge=1, le=100)
    bm25_top_k: int = Field(20, ge=1, le=100)
    final_top_k: int = Field(5, ge=1, le=20)
    rrf_k: int = Field(60, ge=1)
    retrieval_log_path: str = "data/logs/retrievals.jsonl"

    # ---- LLM (free tiers) ------------------------------------------------
    llm_provider: Literal["groq", "gemini", "ollama"] = "groq"
    llm_fallback_provider: Literal["groq", "gemini", "ollama", "none"] = "gemini"
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_rewrite_model: str = "llama-3.1-8b-instant"
    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.0-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    llm_temperature: float = Field(0.1, ge=0, le=1)
    llm_max_output_tokens: int = Field(700, ge=64, le=4096)
    llm_timeout_s: float = 45.0

    # ---- resilience ------------------------------------------------------
    retry_max_attempts: int = Field(4, ge=1, le=10)
    retry_base_delay_s: float = 0.5
    retry_max_delay_s: float = 8.0
    breaker_failure_threshold: int = 5
    breaker_reset_timeout_s: float = 30.0
    llm_requests_per_minute: int = 28          # Groq free tier ~30 RPM
    llm_tokens_per_minute: int = 5500          # stay under free TPM
    api_rate_limit_per_minute: int = 60        # per client IP at the gateway

    # ---- semantic cache --------------------------------------------------
    cache_backend: Literal["redis", "memory"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    cache_similarity_threshold: float = Field(0.92, ge=0.5, le=1.0)
    cache_ttl_s: int = 24 * 3600
    cache_max_entries: int = 5000

    # ---- guardrails ------------------------------------------------------
    use_presidio: bool = False
    use_injection_classifier: bool = False
    injection_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    injection_threshold: float = 0.85

    # ---- service URLs (Cloud Run URLs in GCP) ----------------------------
    retriever_url: HttpUrl = "http://localhost:8001"  # type: ignore[assignment]
    generator_url: HttpUrl = "http://localhost:8002"  # type: ignore[assignment]
    guardrail_url: HttpUrl = "http://localhost:8003"  # type: ignore[assignment]
    use_gcp_id_token: bool = False  # service-to-service auth on Cloud Run
    internal_timeout_s: float = 60.0

    # ---- observability ---------------------------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---- cost table (USD per 1M tokens) ------------------------------------
    # Free tiers cost $0, but we track a "shadow price" so you can see what the
    # same traffic would cost on a paid tier. Override via env as JSON.
    llm_price_per_mtok: dict[str, dict[str, float]] = {
        "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    }
    bill_shadow_cost: bool = False  # False => reported actual cost is 0 (free tier)

    @field_validator("chunk_max_tokens")
    @classmethod
    def _max_gt_min(cls, v: int, info):  # type: ignore[no-untyped-def]
        mn = info.data.get("chunk_min_tokens", 0)
        if v <= mn:
            raise ValueError("chunk_max_tokens must be > chunk_min_tokens")
        return v

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
