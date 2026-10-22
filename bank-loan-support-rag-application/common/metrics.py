"""Model & service monitoring: Prometheus metrics exposed at /metrics on every service.

Tracked:
  * HTTP latency / status per route
  * LLM latency, time-to-first-token, tokens in/out, USD cost, errors (by model)
  * Retrieval latency + candidate counts, semantic cache hits/misses
  * Guardrail blocks / PII redactions, circuit-breaker state, retries
Cloud Run: scrape with Google Managed Prometheus (sidecar) or push the same
numbers to Cloud Monitoring via the structured logs (log-based metrics).
"""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

LAT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32)

HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency",
                         ["service", "route", "method", "status"], buckets=LAT_BUCKETS)
HTTP_INFLIGHT = Gauge("http_requests_inflight", "In-flight requests", ["service"])

LLM_LATENCY = Histogram("llm_request_duration_seconds", "End-to-end LLM call latency",
                        ["model", "operation"], buckets=LAT_BUCKETS)
LLM_TTFT = Histogram("llm_time_to_first_token_seconds", "Streaming time to first token",
                     ["model"], buckets=LAT_BUCKETS)
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["model", "kind"])  # kind=input|output
LLM_COST = Counter("llm_cost_usd_total", "LLM cost in USD (0 on free tier unless shadow billing)", ["model"])
LLM_SHADOW_COST = Counter("llm_shadow_cost_usd_total", "What the traffic would cost on a paid tier", ["model"])
LLM_ERRORS = Counter("llm_errors_total", "LLM errors", ["model", "error_type"])
LLM_FALLBACKS = Counter("llm_fallback_total", "Calls served by the fallback provider", ["from_model", "to_model"])

RETRIEVAL_LATENCY = Histogram("retrieval_duration_seconds", "Retriever stage latency",
                              ["stage"], buckets=LAT_BUCKETS)  # stage=embed|dense|bm25|rrf|total
RETRIEVAL_TOP_SCORE = Histogram("retrieval_top_rrf_score", "RRF score of rank-1 chunk",
                                buckets=(0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035))

CACHE_EVENTS = Counter("semantic_cache_events_total", "Semantic cache lookups", ["result"])  # hit|miss|store
GUARDRAIL_EVENTS = Counter("guardrail_events_total", "Guardrail actions", ["direction", "action"])
BREAKER_STATE = Gauge("circuit_breaker_state", "0=closed 1=half_open 2=open", ["breaker"])
RETRIES = Counter("retries_total", "Retries performed", ["operation"])
THROTTLED = Counter("throttled_requests_total", "Requests rejected by rate limiter", ["service"])


def breaker_listener(name: str, state) -> None:  # type: ignore[no-untyped-def]
    BREAKER_STATE.labels(name).set({"closed": 0, "half_open": 1, "open": 2}[state.value])


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
