"""Prometheus metrics: latency, tokens, cost, cache efficiency, breaker state.

Exposed on /metrics of every service; scraped by Google Managed Prometheus on
Cloud Run (or the bundled Prometheus in docker-compose).

Cost: the configured LLMs are free (Groq free tier, Ollama/vLLM self-hosted), so
`llm_cost_usd_total` is 0 — but we also track `llm_shadow_cost_usd_total`, the
price the same tokens would cost on a paid endpoint, for capacity/budget planning.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# USD per 1M tokens (input, output). Actual cost for free tiers = 0.
SHADOW_PRICES: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "qwen2.5:7b-instruct": (0.10, 0.10),
    "meta-llama/Llama-3.1-8B-Instruct": (0.10, 0.10),
    "rules": (0.0, 0.0),
}
ACTUAL_PRICES: dict[str, tuple[float, float]] = {}  # all free -> 0

REQUESTS = Counter("loanflow_requests_total", "API requests", ["service", "route", "status"])
NODE_LATENCY = Histogram(
    "loanflow_node_latency_seconds", "Latency per graph node / agent", ["node"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)
TOOL_LATENCY = Histogram("loanflow_tool_latency_seconds", "MCP tool latency", ["server", "tool", "ok"])
LLM_LATENCY = Histogram("loanflow_llm_latency_seconds", "LLM call latency", ["provider", "model"])
LLM_TOKENS = Counter("loanflow_llm_tokens_total", "LLM tokens", ["provider", "model", "kind"])
LLM_COST = Counter("loanflow_llm_cost_usd_total", "Actual LLM spend (USD)", ["model"])
LLM_SHADOW_COST = Counter("loanflow_llm_shadow_cost_usd_total", "Equivalent paid-tier cost (USD)", ["model"])
LLM_FALLBACKS = Counter("loanflow_llm_fallbacks_total", "Provider fallbacks", ["from_provider"])
CACHE_EVENTS = Counter("loanflow_cache_events_total", "Cache hits/misses", ["cache", "result"])
GUARDRAIL_EVENTS = Counter("loanflow_guardrail_events_total", "Guardrail triggers", ["type"])
DECISIONS = Counter("loanflow_decisions_total", "Final decisions", ["decision"])
HITL_PENDING = Gauge("loanflow_hitl_pending", "Runs awaiting human review")
RUNS_IN_FLIGHT = Gauge("loanflow_runs_in_flight", "Runs currently executing")
BREAKER_STATE = Gauge("loanflow_circuit_open", "1 if circuit is open", ["breaker"])


def record_llm_usage(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    LLM_TOKENS.labels(provider, model, "prompt").inc(prompt_tokens)
    LLM_TOKENS.labels(provider, model, "completion").inc(completion_tokens)
    pin, pout = ACTUAL_PRICES.get(model, (0.0, 0.0))
    cost = (prompt_tokens * pin + completion_tokens * pout) / 1_000_000
    LLM_COST.labels(model).inc(cost)
    sin, sout = SHADOW_PRICES.get(model, (0.0, 0.0))
    LLM_SHADOW_COST.labels(model).inc((prompt_tokens * sin + completion_tokens * sout) / 1_000_000)
    return cost


@contextmanager
def timed(hist: Histogram, *labels: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        hist.labels(*labels).observe(time.perf_counter() - t0)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
