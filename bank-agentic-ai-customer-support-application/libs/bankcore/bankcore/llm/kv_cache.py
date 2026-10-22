"""KV-cache (prefix cache) optimisation helpers.

Transformer inference stores attention keys/values for every processed token. Servers like
vLLM (``--enable-prefix-caching``, PagedAttention block hashing), SGLang (RadixAttention) and
Ollama/llama.cpp (slot cache + ``keep_alive``) can REUSE the KV blocks of an identical prompt
prefix, skipping prefill for those tokens -> lower TTFT and GPU cost.

To maximise hits, every prompt we build is laid out as:

    [STABLE PREFIX]  system instructions | tool schemas (sorted) | output format
    [DYNAMIC SUFFIX] long-term memory | retrieved facts | short-term notes | user input

The stable prefix is byte-identical across requests for the same agent/version (no timestamps,
no request ids, deterministic JSON). ``PrefixTracker`` measures how often that holds in prod.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

from bankcore.observability.metrics import KV_PREFIX_REUSE


def canonical_tools(tools: list[dict]) -> list[dict]:
    """Sort tools by name and canonicalise JSON so the serialised prefix is deterministic."""
    return json.loads(json.dumps(sorted(tools, key=lambda t: t.get("function", t).get("name", "")),
                                 sort_keys=True))


class PrefixTracker:
    """LRU of recently seen prefix hashes — a hit means the server can likely reuse KV blocks."""

    def __init__(self, capacity: int = 256) -> None:
        self._seen: OrderedDict[str, None] = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def observe(self, prefix_text: str) -> bool:
        h = hashlib.sha256(prefix_text.encode()).hexdigest()
        hit = h in self._seen
        if hit:
            self._seen.move_to_end(h)
            self.hits += 1
        else:
            self._seen[h] = None
            self.misses += 1
            if len(self._seen) > self.capacity:
                self._seen.popitem(last=False)
        KV_PREFIX_REUSE.labels("hit" if hit else "miss").inc()
        return hit

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


prefix_tracker = PrefixTracker()
