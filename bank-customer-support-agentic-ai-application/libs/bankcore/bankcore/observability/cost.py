"""Token, latency and cost accounting via a LangChain callback.

Free models cost $0 in cash, but GPU-seconds / quota are real; configurable *shadow prices*
let FinOps compare self-hosting vs. paid APIs. Usage is aggregated per request and emitted
to Prometheus and Langfuse.
"""
from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from bankcore.config import get_settings
from bankcore.observability.metrics import LLM_COST, LLM_LATENCY, LLM_TOKENS
from bankcore.schemas.common import TokenUsage


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    s = get_settings()
    pin = s.price_per_mtok_input.get(model, s.price_per_mtok_input.get("default", 0.0))
    pout = s.price_per_mtok_output.get(model, s.price_per_mtok_output.get("default", 0.0))
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


class UsageCallback(AsyncCallbackHandler):
    """Collects usage for one request; attach via ``config={"callbacks": [cb]}``."""

    def __init__(self, service: str, purpose: str = "general") -> None:
        self.service, self.purpose = service, purpose
        self.usage = TokenUsage()
        self.cost_usd = 0.0
        self.llm_calls = 0
        self._starts: dict[UUID, float] = {}

    async def on_chat_model_start(self, serialized: dict[str, Any], messages: Any, *, run_id: UUID,
                                  **kwargs: Any) -> None:
        self._starts[run_id] = time.perf_counter()

    async def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID,
                           **kwargs: Any) -> None:
        self._starts[run_id] = time.perf_counter()

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self.llm_calls += 1
        model, inp, out = "unknown", 0, 0
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                um = getattr(msg, "usage_metadata", None) or {}
                inp += int(um.get("input_tokens", 0))
                out += int(um.get("output_tokens", 0))
                meta = getattr(msg, "response_metadata", None) or {}
                model = meta.get("model_name") or meta.get("model") or model
        if not (inp or out) and response.llm_output:
            tu = response.llm_output.get("token_usage", {}) or {}
            inp, out = int(tu.get("prompt_tokens", 0)), int(tu.get("completion_tokens", 0))
            model = response.llm_output.get("model_name", model)
        self.usage = self.usage + TokenUsage(model=model, input_tokens=inp, output_tokens=out)
        cost = estimate_cost(model, inp, out)
        self.cost_usd += cost
        LLM_TOKENS.labels(self.service, model, "input").inc(inp)
        LLM_TOKENS.labels(self.service, model, "output").inc(out)
        LLM_COST.labels(self.service, model).inc(cost)
        started = self._starts.pop(run_id, None)
        if started is not None:
            LLM_LATENCY.labels(self.service, model, self.purpose).observe(time.perf_counter() - started)
