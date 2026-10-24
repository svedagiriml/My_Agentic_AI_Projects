"""Free-LLM client with an ordered fallback chain.

Providers (all free or open-weight, all OpenAI-compatible /chat/completions):
  groq        Llama-3.3-70B / Llama-3.1-8B on Groq's free tier (needs GROQ_API_KEY)
  openrouter  ":free" model variants                              (OPENROUTER_API_KEY)
  ollama      local open-weight models, e.g. qwen2.5:7b-instruct  (no key)
  vllm        self-hosted open-weight model with prefix caching   (no key)
  rules       terminal entry: "no LLM" -> callers use their deterministic policy

Each provider sits behind its own circuit breaker + retry. JSON mode is requested
and the reply is validated with pydantic; one self-repair round is attempted on
schema errors before moving to the next provider.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from loanflow.common.config import get_settings
from loanflow.common.exceptions import LLMError, UpstreamUnavailable
from loanflow.common.logging import get_logger
from loanflow.llm.kv_cache import get_cached_response, prefix_tracker, provider_cache_options, put_cached_response
from loanflow.observability import tracing
from loanflow.observability.metrics import LLM_FALLBACKS, LLM_LATENCY, record_llm_usage
from loanflow.resilience import get_breaker, retry_async

log = get_logger(__name__)
M = TypeVar("M", bound=BaseModel)


class LLMUnavailable(LLMError):
    """Every provider in the chain failed or the chain ended at 'rules'."""
    code = "LLM_UNAVAILABLE"
    retryable = False


@dataclass
class LLMUsage:
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False


@dataclass
class LLMResult:
    parsed: BaseModel
    raw: str
    usage: LLMUsage
    attempts: list[str] = field(default_factory=list)


def _provider_cfg(name: str, small: bool) -> tuple[str, str, dict[str, str]] | None:
    s = get_settings()
    if name == "groq" and s.groq_api_key:
        return ("https://api.groq.com/openai/v1", s.groq_small_model if small else s.groq_model,
                {"Authorization": f"Bearer {s.groq_api_key}"})
    if name == "openrouter" and s.openrouter_api_key:
        return ("https://openrouter.ai/api/v1", s.openrouter_model, {"Authorization": f"Bearer {s.openrouter_api_key}"})
    if name == "ollama":
        return (f"{s.ollama_base_url.rstrip('/')}/v1", s.ollama_model, {})
    if name == "vllm":
        return (s.vllm_base_url.rstrip("/"), s.vllm_model, {})
    return None


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = _JSON_BLOCK.search(text)
    return m.group(0) if m else text


class LLMClient:
    def __init__(self, http: httpx.AsyncClient | None = None):
        self._http = http

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=get_settings().llm_timeout_s)
        return self._http

    async def _call(self, provider: str, base: str, model: str, headers: dict, messages: list[dict]) -> tuple[str, dict]:
        s = get_settings()
        body = {
            "model": model,
            "messages": messages,
            "temperature": s.llm_temperature,
            "max_tokens": s.llm_max_tokens,
            "response_format": {"type": "json_object"},
            **provider_cache_options(provider),
        }
        http = await self._client()

        async def do() -> httpx.Response:
            r = await http.post(f"{base}/chat/completions", json=body, headers=headers)
            if r.status_code == 429 or r.status_code >= 500:
                raise UpstreamUnavailable(f"{provider} HTTP {r.status_code}")
            if r.status_code >= 400:
                raise LLMError(f"{provider} HTTP {r.status_code}: {r.text[:200]}")
            return r

        breaker = get_breaker(f"llm:{provider}", s.cb_failure_threshold, s.cb_reset_timeout_s)
        resp = await breaker.call(lambda: retry_async(do, attempts=s.max_retries, name=f"llm:{provider}"))
        data = resp.json()
        return data["choices"][0]["message"]["content"] or "", data.get("usage", {}) or {}

    async def chat_json(
        self, *, system: str, user: str, schema: type[M], name: str, small: bool = False
    ) -> LLMResult:
        """`system` = static (KV-cacheable) prefix; `user` = dynamic suffix."""
        s = get_settings()
        prefix_tracker.observe(system)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        attempts: list[str] = []
        for provider in s.llm_chain:
            if provider == "rules":
                break
            cfg = _provider_cfg(provider, small)
            if cfg is None:
                attempts.append(f"{provider}:not_configured")
                continue
            base, model, headers = cfg
            cached = await get_cached_response(model, messages)
            if cached:
                try:
                    parsed = schema.model_validate_json(cached["raw"])
                    return LLMResult(parsed, cached["raw"], LLMUsage(provider, model, cached=True), attempts)
                except ValidationError:
                    pass
            t0 = time.perf_counter()
            try:
                with tracing.generation(name, model=model, input=messages) as gen:
                    raw, usage = await self._call(provider, base, model, headers, messages)
                    try:
                        parsed = schema.model_validate_json(_extract_json(raw))
                    except ValidationError as ve:  # one self-repair round
                        repair = messages + [
                            {"role": "assistant", "content": raw},
                            {"role": "user", "content": f"Your JSON was invalid: {str(ve)[:500]}. Return ONLY corrected JSON."},
                        ]
                        raw, usage2 = await self._call(provider, base, model, headers, repair)
                        usage = {k: usage.get(k, 0) + usage2.get(k, 0) for k in ("prompt_tokens", "completion_tokens")}
                        parsed = schema.model_validate_json(_extract_json(raw))
                    pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
                    cost = record_llm_usage(provider, model, pt, ct)
                    gen.update(output=raw, usage_details={"input": pt, "output": ct}, cost_details={"total": cost})
                latency = (time.perf_counter() - t0) * 1000
                LLM_LATENCY.labels(provider, model).observe(latency / 1000)
                await put_cached_response(model, messages, {"raw": raw})
                return LLMResult(parsed, raw, LLMUsage(provider, model, pt, ct, latency, cost), attempts)
            except Exception as exc:  # noqa: BLE001 - fall through to next provider
                attempts.append(f"{provider}:{type(exc).__name__}")
                LLM_FALLBACKS.labels(provider).inc()
                log.warning("llm provider failed", extra={"extra_fields": {"provider": provider, "error": str(exc)[:300]}})
        raise LLMUnavailable("no LLM provider available", details={"attempts": attempts})


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def set_llm(client: LLMClient | None) -> None:
    global _llm
    _llm = client


def dumps_tools(tools: list[dict]) -> str:
    """Deterministic tool catalogue serialisation (stable prefix => KV reuse)."""
    return json.dumps(sorted(tools, key=lambda t: t["name"]), sort_keys=True, indent=None)
