"""LLM factory with provider fallback chain — every model is free / open-weight.

Default chain:  Ollama (local, Qwen2.5 / Llama 3.1)  ->  Groq free tier (Llama 3.3 70B / 3.1 8B)
Optional:       vLLM self-hosted with automatic prefix caching (KV-cache reuse).

Each provider sits behind its own circuit breaker + timeout; ``ResilientLLM`` walks the chain
and raises ``LLMError`` only when every provider failed, so callers can degrade further
(keyword router, cached answer, apology).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from bankcore.config import LLMProvider, Settings, get_settings
from bankcore.errors import CircuitOpenError, LLMError
from bankcore.logging import get_logger
from bankcore.resilience.bulkhead import with_timeout
from bankcore.resilience.circuit_breaker import get_breaker

log = get_logger(__name__)
Role = Literal["supervisor", "subagent", "judge"]

_fake_factory: Callable[[Role], BaseChatModel] | None = None


def set_fake_model_factory(factory: Callable[[Role], BaseChatModel] | None) -> None:
    """Tests inject a scripted model here (LLM_PROVIDER=fake)."""
    global _fake_factory
    _fake_factory = factory


def _model_name(provider: LLMProvider, role: Role, s: Settings) -> str:
    if provider == LLMProvider.GROQ:
        return s.groq_supervisor_model if role in ("supervisor", "judge") else s.groq_subagent_model
    if provider == LLMProvider.VLLM:
        return s.vllm_model
    return {"supervisor": s.supervisor_model, "subagent": s.subagent_model, "judge": s.judge_model}[role]


def build_chat_model(provider: LLMProvider, role: Role, settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()
    name = _model_name(provider, role, s)
    if provider == LLMProvider.FAKE:
        if _fake_factory is None:
            raise LLMError("fake provider selected but no fake model factory registered")
        return _fake_factory(role)
    if provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama
        # keep_alive keeps weights AND the KV cache of the last prompt resident; Ollama reuses the
        # cached prefix when the next prompt shares it (our ContextBuilder keeps prefixes stable).
        return ChatOllama(model=name, base_url=s.ollama_base_url, temperature=s.llm_temperature,
                          keep_alive=s.ollama_keep_alive, num_ctx=s.ollama_num_ctx)
    if provider == LLMProvider.GROQ:
        from langchain_groq import ChatGroq
        if s.groq_api_key is None:
            raise LLMError("GROQ_API_KEY not configured")
        return ChatGroq(model=name, api_key=s.groq_api_key.get_secret_value(), temperature=s.llm_temperature,
                        timeout=s.llm_timeout_s, max_retries=0)
    if provider == LLMProvider.VLLM:
        from langchain_openai import ChatOpenAI
        # vLLM must be started with --enable-prefix-caching (see docker/vllm.Dockerfile).
        return ChatOpenAI(model=name, base_url=s.vllm_base_url, api_key="EMPTY",
                          temperature=s.llm_temperature, timeout=s.llm_timeout_s, max_retries=0)
    raise LLMError(f"unknown provider {provider}")


class ResilientLLM:
    """Provider chain with per-provider circuit breakers and timeouts."""

    def __init__(self, role: Role, settings: Settings | None = None,
                 providers: Sequence[LLMProvider] | None = None) -> None:
        self.role = role
        self.settings = s = settings or get_settings()
        chain = list(providers) if providers else [s.llm_provider, *s.llm_fallback_providers]
        self._models: list[tuple[str, BaseChatModel]] = []
        for p in dict.fromkeys(chain):           # de-dupe, keep order
            try:
                self._models.append((p.value, build_chat_model(p, role, s)))
            except Exception as e:  # noqa: BLE001 — a misconfigured fallback must not break startup
                log.warning("llm provider unavailable", provider=p.value, role=role, error=str(e))
        if not self._models:
            raise LLMError(f"no LLM provider available for role {role}")

    @property
    def primary_name(self) -> str:
        return self._models[0][0]

    async def ainvoke(self, messages: list[BaseMessage], *, tools: list[dict] | None = None,
                      config: dict[str, Any] | None = None, timeout_s: float | None = None) -> AIMessage:
        errors: list[str] = []
        for name, model in self._models:
            breaker = get_breaker(f"llm:{name}:{self.role}")
            runnable = model.bind_tools(tools) if tools else model

            async def _call(r=runnable):
                return await with_timeout(lambda: r.ainvoke(messages, config=config),
                                          timeout_s or self.settings.llm_timeout_s, f"llm:{name}")
            try:
                result = await breaker.call(_call)
                if not isinstance(result, AIMessage):
                    result = AIMessage(content=str(getattr(result, "content", result)))
                return result
            except CircuitOpenError as e:
                errors.append(f"{name}: circuit open")
                log.info("llm provider skipped", provider=name, reason=str(e))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {type(e).__name__}")
                log.warning("llm provider failed, trying fallback", provider=name, error=str(e)[:300])
        raise LLMError(f"all LLM providers failed: {errors}")
