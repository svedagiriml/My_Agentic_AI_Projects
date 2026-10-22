"""Resilient LLM client over free-tier providers.

Providers (all free):
  * groq   – Llama 3.3 70B / Llama 3.1 8B on Groq's free tier (fast, ~30 RPM)
  * gemini – Gemini Flash on Google AI Studio's free tier
  * ollama – any local model, zero cost, no rate limits

Every call goes through, in this order:
  1. throttling   – two token buckets per provider: requests/min and tokens/min,
                    sized just under the free-tier quota, so we queue locally
                    instead of collecting 429s
  2. circuit breaker per provider – after N consecutive failures calls fail fast
  3. exponential backoff + jitter retries (honouring Retry-After on 429)
  4. fallback provider – if the primary breaker is open or retries are exhausted
The SDK's own retries are disabled (max_retries=0) so ours are the only ones.
"""
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from common.config import Settings
from common.cost import estimate_tokens
from common.log import log_event
from common.metrics import LLM_ERRORS, LLM_FALLBACKS, LLM_LATENCY, LLM_TTFT, breaker_listener
from common.resilience import AsyncTokenBucket, CircuitBreaker, CircuitOpenError, resilient_call
from common.schemas import TokenUsage

M = TypeVar("M", bound=BaseModel)
logger = logging.getLogger(__name__)


@dataclass
class Provider:
    name: str
    model_name: str
    llm: BaseChatModel
    breaker: CircuitBreaker
    rpm: AsyncTokenBucket
    tpm: AsyncTokenBucket


def _build_chat_model(provider: str, model: str, s: Settings, streaming: bool = False) -> BaseChatModel:
    common: dict[str, Any] = {"temperature": s.llm_temperature}
    if provider == "groq":
        from langchain_groq import ChatGroq

        if not s.groq_api_key:
            raise ValueError("GROQ_API_KEY missing")
        return ChatGroq(model=model, api_key=s.groq_api_key.get_secret_value(), max_tokens=s.llm_max_output_tokens,
                        timeout=s.llm_timeout_s, max_retries=0, streaming=streaming, **common)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not s.google_api_key:
            raise ValueError("GOOGLE_API_KEY missing")
        return ChatGoogleGenerativeAI(model=model, google_api_key=s.google_api_key.get_secret_value(),
                                      max_output_tokens=s.llm_max_output_tokens, timeout=s.llm_timeout_s,
                                      max_retries=0, **common)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=s.ollama_base_url, num_predict=s.llm_max_output_tokens, **common)
    raise ValueError(f"unknown provider {provider}")


def _model_for(provider: str, s: Settings, purpose: str) -> str:
    if provider == "groq":
        return s.groq_rewrite_model if purpose == "rewrite" else s.groq_model
    if provider == "gemini":
        return s.gemini_model
    return s.ollama_model


def _usage(msg: Any, prompt_text: str, output_text: str) -> TokenUsage:
    um = getattr(msg, "usage_metadata", None) or {}
    if um.get("input_tokens"):
        return TokenUsage(input_tokens=um.get("input_tokens", 0), output_tokens=um.get("output_tokens", 0))
    return TokenUsage(input_tokens=estimate_tokens(prompt_text), output_tokens=estimate_tokens(output_text))


class LLMClient:
    def __init__(self, settings: Settings, purpose: str = "answer",
                 providers: list[Provider] | None = None) -> None:
        self.s = settings
        if providers is not None:
            self.providers = providers
            return
        names = [settings.llm_provider]
        if settings.llm_fallback_provider not in ("none", settings.llm_provider):
            names.append(settings.llm_fallback_provider)
        self.providers = []
        for n in names:
            model = _model_for(n, settings, purpose)
            try:
                llm = _build_chat_model(n, model, settings)
            except Exception as exc:  # noqa: BLE001 - missing key for fallback is fine
                log_event(logger, "provider disabled", logging.WARNING, provider=n, reason=str(exc))
                continue
            self.providers.append(Provider(
                n, model, llm,
                CircuitBreaker(f"llm:{n}:{purpose}", settings.breaker_failure_threshold,
                               settings.breaker_reset_timeout_s, breaker_listener),
                AsyncTokenBucket.per_minute(settings.llm_requests_per_minute),
                AsyncTokenBucket.per_minute(settings.llm_tokens_per_minute),
            ))
        if not self.providers:
            raise RuntimeError("no LLM provider configured (set GROQ_API_KEY or GOOGLE_API_KEY or use ollama)")

    @property
    def primary_model(self) -> str:
        return self.providers[0].model_name

    async def _throttle(self, p: Provider, prompt_text: str) -> None:
        await p.rpm.acquire(1)
        await p.tpm.acquire(estimate_tokens(prompt_text) + self.s.llm_max_output_tokens)

    def _note_fallback(self, i: int, exc: BaseException) -> None:
        p = self.providers[i]
        LLM_ERRORS.labels(p.model_name, type(exc).__name__).inc()
        if i + 1 < len(self.providers):
            LLM_FALLBACKS.labels(p.model_name, self.providers[i + 1].model_name).inc()
        log_event(logger, "llm provider failed", logging.ERROR, provider=p.name, model=p.model_name,
                  error=f"{type(exc).__name__}: {exc}"[:400], will_fallback=i + 1 < len(self.providers))

    # ------------------------------------------------------------------ structured
    async def structured(self, messages: list[BaseMessage], schema: type[M],
                         operation: str) -> tuple[M, TokenUsage, str]:
        """Invoke with Pydantic-enforced structured output -> (parsed, usage, model_name)."""
        prompt_text = "\n".join(str(m.content) for m in messages)
        last_exc: BaseException | None = None
        for i, p in enumerate(self.providers):
            runnable = p.llm.with_structured_output(schema, include_raw=True)

            async def _call(p: Provider = p, runnable: Any = runnable) -> dict[str, Any]:
                await self._throttle(p, prompt_text)
                out = await runnable.ainvoke(messages)
                parsed = out.get("parsed")
                if parsed is None:  # model ignored the tool schema -> try raw JSON once
                    raw = out.get("raw")
                    try:
                        parsed = schema.model_validate_json(str(getattr(raw, "content", "")).strip("` \njson"))
                    except ValidationError as ve:
                        raise ValueError(f"structured output failed validation: {ve}") from ve
                    out["parsed"] = parsed
                return out

            t0 = time.perf_counter()
            try:
                out = await resilient_call(_call, breaker=p.breaker, attempts=self.s.retry_max_attempts,
                                           base_delay=self.s.retry_base_delay_s, max_delay=self.s.retry_max_delay_s,
                                           op_name=f"{p.name}.{operation}")
            except (CircuitOpenError, Exception) as exc:  # noqa: BLE001
                last_exc = exc
                self._note_fallback(i, exc)
                continue
            LLM_LATENCY.labels(p.model_name, operation).observe(time.perf_counter() - t0)
            parsed: M = out["parsed"]
            usage = _usage(out.get("raw"), prompt_text, parsed.model_dump_json())
            return parsed, usage, p.model_name
        raise RuntimeError(f"all LLM providers failed: {last_exc}") from last_exc

    # ------------------------------------------------------------------ streaming
    async def stream(self, messages: list[BaseMessage], usage_out: dict[str, Any]) -> AsyncIterator[str]:
        """Yield text tokens. Retries/fallback happen only before the first token is
        emitted; after that a failure is surfaced to the client as an SSE error."""
        prompt_text = "\n".join(str(m.content) for m in messages)
        last_exc: BaseException | None = None
        for i, p in enumerate(self.providers):
            t0 = time.perf_counter()
            agen = None

            async def _open_and_first(p: Provider = p) -> tuple[Any, Any]:
                nonlocal agen
                await self._throttle(p, prompt_text)
                agen = p.llm.astream(messages).__aiter__()
                first = await agen.__anext__()
                return agen, first

            try:
                agen, first = await resilient_call(_open_and_first, breaker=p.breaker,
                                                   attempts=self.s.retry_max_attempts,
                                                   base_delay=self.s.retry_base_delay_s,
                                                   max_delay=self.s.retry_max_delay_s,
                                                   op_name=f"{p.name}.stream")
            except (CircuitOpenError, Exception) as exc:  # noqa: BLE001
                last_exc = exc
                self._note_fallback(i, exc)
                continue

            LLM_TTFT.labels(p.model_name).observe(time.perf_counter() - t0)
            text_parts: list[str] = []
            final = first
            if first.content:
                text_parts.append(str(first.content))
                yield str(first.content)
            async for chunk in agen:
                final = chunk if getattr(chunk, "usage_metadata", None) else final
                if chunk.content:
                    text_parts.append(str(chunk.content))
                    yield str(chunk.content)
            LLM_LATENCY.labels(p.model_name, "generate_stream").observe(time.perf_counter() - t0)
            usage_out["usage"] = _usage(final, prompt_text, "".join(text_parts))
            usage_out["model"] = p.model_name
            return
        raise RuntimeError(f"all LLM providers failed: {last_exc}") from last_exc
