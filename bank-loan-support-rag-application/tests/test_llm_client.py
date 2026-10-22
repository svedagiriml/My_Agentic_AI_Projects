"""LLMClient: throttling + breaker + retries + provider fallback, with fake chat models."""
from typing import Any

import httpx
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from common.config import Settings
from common.resilience import AsyncTokenBucket, CircuitBreaker
from services.generator.llm_client import LLMClient, Provider


class BrokenModel(GenericFakeChatModel):
    calls: int = 0

    async def _astream(self, *a: Any, **k: Any):  # type: ignore[override]
        object.__setattr__(self, "calls", self.calls + 1)
        req = httpx.Request("POST", "https://api.groq.com")
        raise httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, request=req))
        yield  # pragma: no cover


def _provider(name: str, llm) -> Provider:  # type: ignore[no-untyped-def]
    return Provider(name, f"{name}-model", llm, CircuitBreaker(name, 3, 30),
                    AsyncTokenBucket.per_minute(1000), AsyncTokenBucket.per_minute(1_000_000))


async def test_stream_falls_back_to_secondary_provider():
    s = Settings(retry_max_attempts=2, retry_base_delay_s=0.001, retry_max_delay_s=0.002)
    broken = BrokenModel(messages=iter([]))
    good = GenericFakeChatModel(messages=iter([AIMessage("Foreclosure is nil after 24 months [S1].")]))
    client = LLMClient(s, providers=[_provider("groq", broken), _provider("gemini", good)])
    box: dict = {}
    text = "".join([t async for t in client.stream([HumanMessage("q")], box)])
    assert "nil after 24 months" in text
    assert broken.calls == 2                    # retried, then fell back
    assert box["model"] == "gemini-model" and box["usage"].output_tokens > 0
