"""KV-cache-aware inference optimisation.

Transformer inference spends most of its prefill time computing key/value tensors
for the prompt. If consecutive requests share a *byte-identical prefix*, the server
can reuse those KV blocks instead of recomputing them:

  * vLLM      : Automatic Prefix Caching   (`--enable-prefix-caching`, on in docker-compose)
  * llama.cpp : `cache_prompt: true` per request (slot KV reuse)
  * Ollama    : reuses the slot's KV cache for a matching prefix while the model stays
                loaded -> we send `keep_alive` so the model (and its cache) is not evicted
  * Groq      : server-side, transparent

What the application must do to benefit is make prefixes *stable*:
  1. Static, cacheable content first (system instructions, tool catalogue, output
     format) and dynamic content (user input, retrieved facts, notes) last.
     -> enforced by context/builder.py ordering.
  2. No timestamps / random ids in the static prefix.
  3. Deterministic serialisation of tool schemas (sorted keys).

This module (a) computes the static-prefix fingerprint for metrics so we can see the
expected KV reuse rate, (b) adds the provider-specific request flags and (c) offers an
exact-match response cache for temperature-0 calls (skips inference entirely).
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

from loanflow.common.config import get_settings
from loanflow.memory.kv import get_kv
from loanflow.observability.metrics import CACHE_EVENTS


class PrefixTracker:
    """Tracks how often the static prompt prefix repeats (=> KV prefix reuse on the server)."""

    def __init__(self, maxsize: int = 512):
        self._seen: OrderedDict[str, int] = OrderedDict()
        self.maxsize = maxsize
        self.reuse = 0
        self.total = 0

    def observe(self, static_prefix: str) -> str:
        h = hashlib.sha256(static_prefix.encode()).hexdigest()[:16]
        self.total += 1
        if h in self._seen:
            self.reuse += 1
            self._seen.move_to_end(h)
            CACHE_EVENTS.labels("kv_prefix", "hit").inc()
        else:
            CACHE_EVENTS.labels("kv_prefix", "miss").inc()
            self._seen[h] = 0
            if len(self._seen) > self.maxsize:
                self._seen.popitem(last=False)
        return h

    @property
    def reuse_ratio(self) -> float:
        return self.reuse / self.total if self.total else 0.0


prefix_tracker = PrefixTracker()


def provider_cache_options(provider: str) -> dict[str, Any]:
    """Extra request-body fields that enable/retain KV caching per provider."""
    s = get_settings()
    if provider == "ollama":
        # keep model + KV slot resident; fixed num_ctx avoids cache invalidation on resize
        return {"keep_alive": s.ollama_keep_alive, "options": {"num_ctx": 8192}}
    if provider == "llamacpp":
        return {"cache_prompt": True}
    return {}  # vLLM APC is server-side; Groq/OpenRouter transparent


def response_cache_key(model: str, messages: list[dict[str, str]]) -> str:
    body = json.dumps({"m": model, "msgs": messages}, sort_keys=True, separators=(",", ":"))
    return "llmcache:" + hashlib.sha256(body.encode()).hexdigest()


async def get_cached_response(model: str, messages: list[dict[str, str]]) -> dict | None:
    if get_settings().llm_temperature > 0:
        return None  # only deterministic calls are safe to cache
    hit = await get_kv().get(response_cache_key(model, messages))
    CACHE_EVENTS.labels("llm_response", "hit" if hit else "miss").inc()
    return hit


async def put_cached_response(model: str, messages: list[dict[str, str]], response: dict, ttl_s: int = 3600) -> None:
    if get_settings().llm_temperature == 0:
        await get_kv().set(response_cache_key(model, messages), response, ttl_s=ttl_s)
