"""Centralised, environment-driven configuration.

Every service (supervisor, each subagent, each MCP server) reads the same
Settings object; only the env vars differ per Cloud Run service.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from pydantic import BaseModel, Field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in {"1", "true", "yes", "on"}


AGENT_NAMES = [
    "fetch_document",
    "validate_document",
    "check_credit",
    "assess_risk",
    "check_compliance",
    "compile_report",
    "compile_rejection",
]

MCP_SERVER_NAMES = ["document", "credit_bureau", "risk_engine", "compliance", "reporting"]


class Settings(BaseModel):
    env: str = Field(default_factory=lambda: _env("APP_ENV", "local"))  # local | test | staging | prod
    service_name: str = Field(default_factory=lambda: _env("SERVICE_NAME", "supervisor"))
    log_level: str = Field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # --- LLMs (all free / open-weight) -----------------------------------
    # Ordered fallback chain. "rules" = deterministic offline planner (no LLM).
    llm_chain: list[str] = Field(
        default_factory=lambda: [p.strip() for p in _env("LLM_CHAIN", "groq,ollama,rules").split(",") if p.strip()]
    )
    groq_api_key: str = Field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str = Field(default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_small_model: str = Field(default_factory=lambda: _env("GROQ_SMALL_MODEL", "llama-3.1-8b-instant"))
    ollama_base_url: str = Field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = Field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    vllm_base_url: str = Field(default_factory=lambda: _env("VLLM_BASE_URL", "http://localhost:8008/v1"))
    vllm_model: str = Field(default_factory=lambda: _env("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    openrouter_api_key: str = Field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    openrouter_model: str = Field(
        default_factory=lambda: _env("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    )
    llm_timeout_s: float = Field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 30.0))
    llm_temperature: float = Field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_max_tokens: int = Field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1024))
    ollama_keep_alive: str = Field(default_factory=lambda: _env("OLLAMA_KEEP_ALIVE", "30m"))  # keeps KV cache warm
    embedding_backend: str = Field(default_factory=lambda: _env("EMBEDDING_BACKEND", "hash"))  # hash | sentence_transformers
    embedding_model: str = Field(default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    embedding_dim: int = Field(default_factory=lambda: _env_int("EMBEDDING_DIM", 384))

    # --- Memory --------------------------------------------------------
    redis_url: str = Field(default_factory=lambda: _env("REDIS_URL", ""))  # empty => in-memory fallback
    postgres_dsn: str = Field(default_factory=lambda: _env("POSTGRES_DSN", ""))  # empty => in-memory fallback
    short_term_ttl_s: int = Field(default_factory=lambda: _env_int("SHORT_TERM_TTL_S", 86_400))
    semantic_cache_threshold: float = Field(default_factory=lambda: _env_float("SEMANTIC_CACHE_THRESHOLD", 0.92))
    semantic_cache_ttl_s: int = Field(default_factory=lambda: _env_int("SEMANTIC_CACHE_TTL_S", 3_600))

    # --- Transport: how supervisor reaches agents and agents reach MCP ---
    # "inproc" for local dev/tests; "http" for docker-compose / Cloud Run.
    transport_mode: str = Field(default_factory=lambda: _env("TRANSPORT_MODE", "inproc"))
    agent_urls: dict[str, str] = Field(default_factory=lambda: json.loads(_env("AGENT_URLS", "{}")))
    mcp_urls: dict[str, str] = Field(default_factory=lambda: json.loads(_env("MCP_URLS", "{}")))
    use_gcp_id_tokens: bool = Field(default_factory=lambda: _env_bool("USE_GCP_ID_TOKENS", False))

    # --- Security ------------------------------------------------------
    jwt_secret: str = Field(default_factory=lambda: _env("JWT_SECRET", "dev-only-change-me-32-bytes-minimum!!"))
    jwt_issuer: str = Field(default_factory=lambda: _env("JWT_ISSUER", "loanflow-auth"))
    jwt_audience: str = Field(default_factory=lambda: _env("JWT_AUDIENCE", "loanflow-api"))
    message_hmac_key: str = Field(default_factory=lambda: _env("MESSAGE_HMAC_KEY", "dev-hmac-key-change-me"))
    internal_service_token: str = Field(default_factory=lambda: _env("INTERNAL_SERVICE_TOKEN", "dev-internal-token"))
    rate_limit_per_minute: int = Field(default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 60))

    # --- Resilience ----------------------------------------------------
    agent_timeout_s: float = Field(default_factory=lambda: _env_float("AGENT_TIMEOUT_S", 60.0))
    tool_timeout_s: float = Field(default_factory=lambda: _env_float("TOOL_TIMEOUT_S", 15.0))
    max_retries: int = Field(default_factory=lambda: _env_int("MAX_RETRIES", 3))
    cb_failure_threshold: int = Field(default_factory=lambda: _env_int("CB_FAILURE_THRESHOLD", 5))
    cb_reset_timeout_s: float = Field(default_factory=lambda: _env_float("CB_RESET_TIMEOUT_S", 30.0))
    max_concurrent_runs: int = Field(default_factory=lambda: _env_int("MAX_CONCURRENT_RUNS", 20))
    worker_enabled: bool = Field(default_factory=lambda: _env_bool("WORKER_ENABLED", True))  # consume job queue in this process
    react_max_steps: int = Field(default_factory=lambda: _env_int("REACT_MAX_STEPS", 6))
    supervisor_max_steps: int = Field(default_factory=lambda: _env_int("SUPERVISOR_MAX_STEPS", 15))

    # --- Business policy (kept in config so risk/compliance can tune) ---
    hitl_amount_threshold_inr: float = Field(default_factory=lambda: _env_float("HITL_AMOUNT_THRESHOLD_INR", 5_000_000))
    min_credit_score: int = Field(default_factory=lambda: _env_int("MIN_CREDIT_SCORE", 650))
    max_dti: float = Field(default_factory=lambda: _env_float("MAX_DTI", 0.50))

    # --- Observability -------------------------------------------------
    langfuse_public_key: str = Field(default_factory=lambda: _env("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str = Field(default_factory=lambda: _env("LANGFUSE_SECRET_KEY"))
    langfuse_host: str = Field(default_factory=lambda: _env("LANGFUSE_HOST", "http://localhost:3000"))

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    def validate_for_prod(self) -> None:
        """Fail fast on insecure defaults in production."""
        if not self.is_prod:
            return
        insecure = []
        if "dev" in self.jwt_secret or len(self.jwt_secret) < 32:
            insecure.append("JWT_SECRET")
        if "dev" in self.message_hmac_key:
            insecure.append("MESSAGE_HMAC_KEY")
        if "dev" in self.internal_service_token:
            insecure.append("INTERNAL_SERVICE_TOKEN")
        if not self.redis_url:
            insecure.append("REDIS_URL")
        if not self.postgres_dsn:
            insecure.append("POSTGRES_DSN")
        if insecure:
            raise RuntimeError(f"Insecure/missing production settings: {insecure}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.validate_for_prod()
    return s


def reset_settings_cache() -> None:
    get_settings.cache_clear()
