"""Prometheus metrics (scraped by Managed Prometheus / Cloud Monitoring on Cloud Run)."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

_LAT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60)

REQUEST_LATENCY = Histogram("bank_request_latency_seconds", "End-to-end HTTP latency",
                            ["service", "route", "status"], buckets=_LAT_BUCKETS)
LLM_LATENCY = Histogram("bank_llm_latency_seconds", "LLM call latency", ["service", "model", "purpose"],
                        buckets=_LAT_BUCKETS)
LLM_TOKENS = Counter("bank_llm_tokens_total", "LLM tokens", ["service", "model", "direction"])
LLM_COST = Counter("bank_llm_cost_usd_total", "Estimated LLM cost (shadow price for free models)",
                   ["service", "model"])
TOOL_CALLS = Counter("bank_tool_calls_total", "MCP tool calls", ["agent", "tool", "outcome"])
TOOL_LATENCY = Histogram("bank_tool_latency_seconds", "MCP tool latency", ["agent", "tool"], buckets=_LAT_BUCKETS)
AGENT_TASKS = Counter("bank_agent_tasks_total", "Sub-agent task outcomes", ["agent", "status"])
ROUTING_CACHE = Counter("bank_routing_cache_total", "Semantic routing cache lookups", ["result"])
GUARDRAIL_EVENTS = Counter("bank_guardrail_events_total", "Guardrail interventions", ["stage", "reason"])
HITL_EVENTS = Counter("bank_hitl_events_total", "Human-in-the-loop events", ["agent", "event"])
CIRCUIT_STATE = Gauge("bank_circuit_state", "Circuit breaker state 0=closed 1=half-open 2=open", ["name"])
MEMORY_OPS = Counter("bank_memory_ops_total", "Memory operations", ["store", "op", "outcome"])
KV_PREFIX_REUSE = Counter("bank_kv_prefix_total", "Prompt prefix reuse (KV-cache friendliness)", ["result"])
