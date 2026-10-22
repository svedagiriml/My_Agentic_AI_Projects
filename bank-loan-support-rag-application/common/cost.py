"""Token cost accounting."""
from __future__ import annotations

from common.config import Settings
from common.metrics import LLM_COST, LLM_SHADOW_COST, LLM_TOKENS
from common.schemas import TokenUsage


def compute_cost(model: str, usage: TokenUsage, settings: Settings) -> tuple[float, float]:
    """Return (actual_cost_usd, shadow_cost_usd) and record metrics."""
    price = settings.llm_price_per_mtok.get(model, {"input": 0.0, "output": 0.0})
    shadow = usage.input_tokens / 1e6 * price["input"] + usage.output_tokens / 1e6 * price["output"]
    actual = shadow if settings.bill_shadow_cost else 0.0
    LLM_TOKENS.labels(model, "input").inc(usage.input_tokens)
    LLM_TOKENS.labels(model, "output").inc(usage.output_tokens)
    LLM_SHADOW_COST.labels(model).inc(shadow)
    LLM_COST.labels(model).inc(actual)
    return round(actual, 8), round(shadow, 8)


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer-free estimate (~4 chars/token for English) used for throttling."""
    return max(1, len(text) // 4)
