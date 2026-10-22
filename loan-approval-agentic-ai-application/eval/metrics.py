"""Evaluation metrics for the multi-agent loan system.

Families
  answer quality      decision accuracy, required-reason coverage, rubric checks, optional LLM-as-judge
  trajectory          exact match, LCS similarity, node precision/recall (agent routing quality)
  tool selection      per-node precision / recall / F1 of tools called vs expected
  hallucination       ungrounded numbers in the narrative, fact mismatches vs tool outputs
  retrieval           precision@k, recall@k, MRR, nDCG@k on long-term memory
  efficiency          latency p50/p95, LLM calls, tokens, (shadow) cost
  human agreement     accuracy vs human labels, Cohen's kappa (from HITL review)
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any

from loanflow.security.guardrails import contains_pii, ungrounded_numbers


# ------------------------------------------------------------------ trajectory
def lcs_len(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            dp[i + 1][j + 1] = dp[i][j] + 1 if x == y else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def trajectory_scores(actual: list[str], expected: list[str]) -> dict[str, float]:
    lcs = lcs_len(actual, expected)
    a, e = set(actual), set(expected)
    return {
        "trajectory_exact": float(actual == expected),
        "trajectory_lcs": lcs / max(len(actual), len(expected), 1),
        "node_precision": len(a & e) / len(a) if a else float(not e),
        "node_recall": len(a & e) / len(e) if e else 1.0,
    }


# ------------------------------------------------------------------ tools
def prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def tool_selection_scores(tool_calls: list[dict[str, Any]], expected: dict[str, list[str]]) -> dict[str, float]:
    called: dict[str, set[str]] = {}
    for t in tool_calls:
        called.setdefault(t["node"], set()).add(t["tool"])
    ps, rs, fs = [], [], []
    for node in set(expected) | set(called):
        p, r, f = prf(called.get(node, set()), set(expected.get(node, [])))
        ps.append(p), rs.append(r), fs.append(f)
    ok = [t["ok"] for t in tool_calls]
    return {
        "tool_precision": statistics.mean(ps) if ps else 1.0,
        "tool_recall": statistics.mean(rs) if rs else 1.0,
        "tool_f1": statistics.mean(fs) if fs else 1.0,
        "tool_success_rate": sum(ok) / len(ok) if ok else 1.0,
        "redundant_tool_calls": float(sum(c - 1 for c in Counter((t["node"], t["tool"]) for t in tool_calls).values() if c > 1)),
    }


# ------------------------------------------------------------------ hallucination
def hallucination_scores(decision: dict[str, Any] | None, results: dict[str, Any], expected_facts: dict[str, Any]) -> dict[str, float]:
    if not decision:
        return {"hallucinated_numbers": 0.0, "fact_mismatch": 0.0, "pii_leak": 0.0}
    grounding = str(results)
    text = " ".join([decision.get("summary", ""), *decision.get("reasons", []), *decision.get("conditions", [])])
    ungrounded = ungrounded_numbers(text, grounding + str(decision.get("approved_amount_inr")) + str(decision.get("interest_rate_pct")))
    mismatches = sum(1 for k, v in expected_facts.items() if decision.get(k) is not None and decision.get(k) != v)
    # tool-grounding: decision fields must equal what the tools returned
    cr = (results.get("check_credit") or {}).get("credit_score")
    if cr is not None and decision.get("credit_score") not in (None, cr):
        mismatches += 1
    return {"hallucinated_numbers": float(len(ungrounded)), "fact_mismatch": float(mismatches),
            "pii_leak": float(contains_pii(text))}


# ------------------------------------------------------------------ answer quality
def answer_quality(decision: dict[str, Any] | None, exp: dict[str, Any]) -> dict[str, float]:
    if exp.get("decision") is None:
        return {"decision_correct": float(decision is None), "reason_coverage": 1.0, "rubric": 1.0}
    if not decision:
        return {"decision_correct": 0.0, "reason_coverage": 0.0, "rubric": 0.0}
    reasons = " ".join(decision.get("reasons", []))
    req = exp.get("reasons_contain", [])
    coverage = sum(1 for r in req if r.lower() in reasons.lower()) / len(req) if req else 1.0
    rubric_checks = [
        bool(decision.get("summary")),
        len(decision.get("reasons", [])) >= 1,
        bool(decision.get("audit_trail")),
        bool(re.fullmatch(r"[a-f0-9]{64}", decision.get("report_sha256") or "")),
        (decision["decision"] != "APPROVED") or (decision.get("approved_amount_inr") and decision.get("interest_rate_pct")),
        (decision["decision"] != "REJECTED") or not decision.get("approved_amount_inr"),
        not exp.get("hitl") or bool(decision.get("human_review")),
    ]
    return {"decision_correct": float(decision["decision"] == exp["decision"]), "reason_coverage": coverage,
            "rubric": sum(map(bool, rubric_checks)) / len(rubric_checks)}


# ------------------------------------------------------------------ retrieval
def retrieval_scores(retrieved: list[str], relevant: list[str], k: int) -> dict[str, float]:
    rel = [any(r.lower() in doc.lower() for r in relevant) for doc in retrieved[:k]]
    hits = sum(rel)
    first = next((i for i, x in enumerate(rel) if x), None)
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(rel) if x)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    found = {r for r in relevant if any(r.lower() in d.lower() for d in retrieved[:k])}
    return {
        f"precision@{k}": hits / k,
        f"recall@{k}": len(found) / len(relevant) if relevant else 1.0,
        "mrr": 1 / (first + 1) if first is not None else 0.0,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        "hit": float(hits > 0),
    }


# ------------------------------------------------------------------ human agreement
def cohen_kappa(a: list[str], b: list[str]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    labels = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / len(a)
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[lab] / len(a)) * (cb[lab] / len(b)) for lab in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    idx = min(len(v) - 1, max(0, math.ceil(p / 100 * len(v)) - 1))
    return v[idx]
