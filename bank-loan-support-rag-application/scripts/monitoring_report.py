"""Print a quick cost / latency / cache report by scraping each service's /metrics.

    python scripts/monitoring_report.py --gateway http://localhost:8000 --generator http://localhost:8002 \
        --retriever http://localhost:8001
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import httpx
from prometheus_client.parser import text_string_to_metric_families


def scrape(url: str) -> dict[str, list]:
    out: dict[str, list] = defaultdict(list)
    for fam in text_string_to_metric_families(httpx.get(f"{url}/metrics", timeout=10).text):
        for s in fam.samples:
            out[s.name].append(s)
    return out


def quantile(samples: list, q: float, **labels: str) -> float | None:
    buckets = sorted(((float(s.labels["le"]), s.value) for s in samples
                      if all(s.labels.get(k) == v for k, v in labels.items())), key=lambda x: x[0])
    agg: dict[float, float] = defaultdict(float)
    for le, v in buckets:
        agg[le] += v
    items = sorted(agg.items())
    if not items or items[-1][1] == 0:
        return None
    target = q * items[-1][1]
    prev_le, prev_c = 0.0, 0.0
    for le, c in items:
        if c >= target:
            if le == float("inf"):
                return prev_le
            return prev_le + (le - prev_le) * ((target - prev_c) / max(c - prev_c, 1e-9))
        prev_le, prev_c = le, c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://localhost:8000")
    ap.add_argument("--generator", default="http://localhost:8002")
    ap.add_argument("--retriever", default="http://localhost:8001")
    a = ap.parse_args()

    gw, gen, ret = scrape(a.gateway), scrape(a.generator), scrape(a.retriever)
    p50 = quantile(gw["http_request_duration_seconds_bucket"], 0.5, route="/v1/chat")
    p95 = quantile(gw["http_request_duration_seconds_bucket"], 0.95, route="/v1/chat")
    ttft = quantile(gen["llm_time_to_first_token_seconds_bucket"], 0.95)
    tokens = defaultdict(float)
    for s in gen["llm_tokens_total"]:
        tokens[(s.labels["model"], s.labels["kind"])] += s.value
    cost = sum(s.value for s in gen["llm_cost_usd_total"])
    shadow = sum(s.value for s in gen["llm_shadow_cost_usd_total"])
    cache = {s.labels["result"]: s.value for s in gw["semantic_cache_events_total"]}
    hits, misses = cache.get("hit", 0), cache.get("miss", 0)
    ret_p95 = quantile(ret["retrieval_duration_seconds_bucket"], 0.95, stage="total")

    def fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}s"

    print("=== Bank RAG monitoring report ===")
    print(f"gateway /v1/chat latency  p50={fmt(p50)}  p95={fmt(p95)}  (stream: time to last byte header)")
    print(f"LLM time-to-first-token   p95={fmt(ttft)}")
    print(f"retrieval total           p95={fmt(ret_p95)}")
    for (model, kind), v in sorted(tokens.items()):
        print(f"tokens {kind:<6} {model:<28} {int(v):>10}")
    print(f"LLM cost actual=${cost:.4f}  shadow (paid-tier equivalent)=${shadow:.4f}")
    rate = hits / (hits + misses) if hits + misses else 0
    print(f"semantic cache hit rate   {rate:.1%}  ({int(hits)} hits / {int(misses)} misses)")


if __name__ == "__main__":
    main()
