"""Centralised, validated configuration (12-factor: everything comes from env vars).

In Cloud Run, secrets (JWT keys, HMAC secret, DB passwords, Langfuse keys) are
mounted from Secret Manager as env vars — never baked into images.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    GROQ = "groq"        # free tier, hosted open-weight models (Llama 3.x)
    OLLAMA = "ollama"    # free, local open-weight models
    VLLM = "vllm"        # free, self-hosted open-weight models w/ prefix (KV) caching
    FAKE = "fake"        # deterministic scripted model for tests


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- service identity -------------------------------------------------
    environment: Environment = Environment.LOCAL
    service_name: str = "bank-agentic"
    log_level: str = "INFO"

    # ---- LLMs (all free / open weights) -----------------------------------
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_fallback_providers: list[LLMProvider] = Field(default_factory=lambda: [LLMProvider.GROQ])
    supervisor_model: str = "qwen2.5:7b-instruct"          # CoT planner ("brain")
    subagent_model: str = "qwen2.5:7b-instruct"            # ReAct workers
    judge_model: str = "qwen2.5:7b-instruct"               # LLM-as-judge for eval
    groq_api_key: SecretStr | None = None
    groq_supervisor_model: str = "llama-3.3-70b-versatile"
    groq_subagent_model: str = "llama-3.1-8b-instant"
    ollama_base_url: str = "http://localhost:11434"
    ollama_keep_alive: str = "30m"     # keep weights + KV cache resident between calls
    ollama_num_ctx: int = 8192
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    # ---- embeddings (free, local) ----------------------------------------
    embedding_backend: str = "sentence-transformers"   # sentence-transformers | ollama | hash
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # ---- data stores -----------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    postgres_dsn: SecretStr = SecretStr("postgresql://bank:bank@localhost:5432/bank")
    short_term_ttl_s: int = 60 * 60 * 24         # conversation state lives 24h in Redis
    short_term_max_turns: int = 20

    # ---- semantic cache --------------------------------------------------
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.92       # cosine similarity for a routing-cache hit
    semantic_cache_ttl_s: int = 60 * 60 * 6

    # ---- security --------------------------------------------------------
    jwt_secret: SecretStr = SecretStr("change-me-dev-only-secret-change-me-32b")
    jwt_issuer: str = "bank-idp"
    jwt_audience: str = "bank-support-api"
    jwt_algorithm: str = "HS256"
    handoff_hmac_secret: SecretStr = SecretStr("change-me-handoff-secret-32-bytes!!")
    delegation_secret: SecretStr = SecretStr("change-me-delegation-secret-32bytes")
    approval_secret: SecretStr = SecretStr("change-me-approval-secret-32-bytes!")
    service_auth_mode: str = "local"             # local (HS256 service JWT) | gcp (Cloud Run ID tokens)
    handoff_max_skew_s: int = 120                # replay window for signed envelopes

    # ---- limits / resilience ---------------------------------------------
    rate_limit_per_minute: int = 30
    rate_limit_per_day: int = 500
    request_deadline_s: float = 60.0
    subagent_timeout_s: float = 25.0
    mcp_timeout_s: float = 10.0
    cb_failure_threshold: int = 5
    cb_recovery_s: float = 30.0
    bulkhead_max_concurrency: int = 20
    react_max_iterations: int = 6

    # ---- HITL thresholds -------------------------------------------------
    hitl_transfer_threshold: float = 1000.0      # transfers >= this need staff approval
    hitl_fund_subscription_threshold: float = 5000.0
    hitl_atm_limit_threshold: float = 2000.0

    # ---- service discovery (Cloud Run URLs) ------------------------------
    supervisor_url: str = "http://localhost:8000"
    agent_urls: dict[str, str] = Field(default_factory=lambda: {
        "account": "http://localhost:8101",
        "payments": "http://localhost:8102",
        "cards": "http://localhost:8103",
        "fraud": "http://localhost:8104",
        "loans": "http://localhost:8105",
        "investments": "http://localhost:8106",
        "knowledge": "http://localhost:8107",
        "compliance": "http://localhost:8108",
    })
    mcp_urls: dict[str, str] = Field(default_factory=lambda: {
        "account": "http://localhost:9101/mcp/",
        "payments": "http://localhost:9102/mcp/",
        "cards": "http://localhost:9103/mcp/",
        "fraud": "http://localhost:9104/mcp/",
        "loans": "http://localhost:9105/mcp/",
        "investments": "http://localhost:9106/mcp/",
        "knowledge": "http://localhost:9107/mcp/",
        "compliance": "http://localhost:9108/mcp/",
    })

    # ---- async messaging -------------------------------------------------
    event_bus_backend: str = "redis"             # redis (Streams) | pubsub (GCP)
    gcp_project: str | None = None

    # ---- observability ---------------------------------------------------
    langfuse_enabled: bool = True
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "http://localhost:3000"
    # shadow prices (USD / 1M tokens) to estimate cost even for free models
    price_per_mtok_input: dict[str, float] = Field(default_factory=lambda: {"default": 0.0})
    price_per_mtok_output: dict[str, float] = Field(default_factory=lambda: {"default": 0.0})

    # ---- sub-agent identity (set per Cloud Run service) ------------------
    agent_name: str = "account"

    @field_validator("jwt_secret", "handoff_hmac_secret", "delegation_secret", "approval_secret")
    @classmethod
    def _min_secret_len(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 32:
            raise ValueError("secrets must be >= 32 bytes")
        return v

    @property
    def is_prod(self) -> bool:
        return self.environment in (Environment.STAGING, Environment.PRODUCTION)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.is_prod and "change-me" in s.jwt_secret.get_secret_value():
        raise RuntimeError("Refusing to start in production with development secrets")
    return s
