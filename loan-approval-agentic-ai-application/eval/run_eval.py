"""Golden-dataset evaluation runner (also the CI quality gate).

  python -m eval.run_eval                       # offline, deterministic (LLM_CHAIN=rules)
  LLM_CHAIN=groq,ollama,rules python -m eval.run_eval --judge   # with live LLMs + LLM-as-judge

For each golden case the full multi-agent graph is executed; HITL interrupts are answered
with the human decision recorded in the dataset (simulated underwriter). Writes
eval/reports/report.json + report.md and exits non-zero if any threshold fails.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("LLM_CHAIN", "rules")
os.environ.setdefault("APP_ENV", "test")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from eval.metrics import (  # noqa: E402
    answer_quality,
    hallucination_scores,
    percentile,
    retrieval_scores,
    tool_selection_scores,
    trajectory_scores,
)
from loanflow.memory.long_term import get_long_term_memory, seed_long_term_memory  # noqa: E402
from loanflow.supervisor.graph import build_graph  # noqa: E402
from loanflow.supervisor.runner import LoanOrchestrator  # noqa: E402

HERE = Path(__file__).parent
THRESHOLDS = {
    "decision_accuracy": 0.95, "trajectory_exact": 0.90, "tool_f1": 0.95, "reason_coverage": 0.90,
    "rubric": 0.95, "hallucinated_numbers_total": 0, "fact_mismatch_total": 0, "pii_leaks": 0,
    "retrieval_hit_rate": 0.85, "retrieval_mrr": 0.70,
}


class JudgeScore(BaseModel):
    reasoning: str = Field(max_length=1500)
    faithfulness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)


async def llm_judge(decision: dict, results: dict) -> dict | None:
    from loanflow.llm.client import LLMUnavailable, get_llm

    system = ("You are a strict credit-audit reviewer. Score the loan decision 1-5 for faithfulness to the facts, "
              "completeness of reasons and clarity. Return JSON with reasoning, faithfulness, completeness, clarity.")
    try:
        res = await get_llm().chat_json(system=system, user=json.dumps({"facts": results, "decision": decision}, default=str)[:12000],
                                        schema=JudgeScore, name="eval.judge")
        return res.parsed.model_dump()  # type: ignore[union-attr]
    except LLMUnavailable:
        return None


async def run_case(orch: LoanOrchestrator, case: dict, judge: bool) -> dict:
    run_id = f"run_eval_{case['id'].lower()}_{int(time.time() * 1000) % 10_000_000}"
    t0 = time.perf_counter()
    st = await orch.execute(run_id, case["applicant_id"], case["document_id"], "eval-officer")
    hitl_triggered = st.status == "awaiting_human"
    if hitl_triggered:
        hr = case.get("human_review") or {"decision": "REJECTED", "reviewer_notes": "no golden human decision"}
        st = await orch.resume(run_id, hr, "eval-underwriter")
    latency = (time.perf_counter() - t0) * 1000
    audit = await orch.audit(run_id)
    snap = await orch.graph.aget_state(orch._cfg(run_id))
    results = snap.values.get("results", {})
    decision = st.decision.model_dump(mode="json") if st.decision else None
    exp = case["expected"]
    row = {
        "id": case["id"], "status": st.status, "expected_status": exp["status"],
        "decision": decision and decision["decision"], "expected_decision": exp["decision"],
        "trajectory": st.trajectory, "expected_trajectory": exp["trajectory"],
        "hitl_triggered": hitl_triggered, "hitl_expected": exp["hitl"],
        "error_code": st.error.code if st.error else None, "latency_ms": round(latency, 1),
        "llm_calls": audit["metrics"].get("llm_calls", 0),
        "tokens": audit["metrics"].get("prompt_tokens", 0) + audit["metrics"].get("completion_tokens", 0),
        "cost_usd": audit["metrics"].get("cost_usd", 0.0),
        "routing_sources": [r["source"] for r in audit["routing_log"]],
        **trajectory_scores(st.trajectory, exp["trajectory"]),
        **tool_selection_scores(audit["tool_calls"], exp["tools"]),
        **hallucination_scores(decision, results, exp.get("facts", {})),
        **answer_quality(decision, exp),
        "status_correct": float(st.status == exp["status"] and (exp.get("error_code") in (None, st.error and st.error.code))),
        "hitl_correct": float(hitl_triggered == exp["hitl"]),
        "final_summary": decision and decision.get("summary"),
    }
    if judge and decision:
        row["judge"] = await llm_judge(decision, results)
    return row


async def run_retrieval(cases: list[dict], k: int = 3) -> list[dict]:
    ltm = get_long_term_memory()
    rows = []
    for c in cases:
        hits = await ltm.recall(c["query"], kinds=c["kinds"], namespaces=["global"], k=k, min_score=0.0)
        rows.append({"id": c["id"], "query": c["query"], "retrieved": [h.record.content for h in hits],
                     **retrieval_scores([h.record.content for h in hits], c["relevant"], k)})
    return rows


def aggregate(rows: list[dict], rrows: list[dict]) -> dict:
    mean = lambda key: round(statistics.mean(r[key] for r in rows), 4)  # noqa: E731
    lat = [r["latency_ms"] for r in rows]
    return {
        "cases": len(rows),
        "decision_accuracy": mean("decision_correct"), "status_accuracy": mean("status_correct"),
        "hitl_accuracy": mean("hitl_correct"), "trajectory_exact": mean("trajectory_exact"),
        "trajectory_lcs": mean("trajectory_lcs"), "node_precision": mean("node_precision"), "node_recall": mean("node_recall"),
        "tool_precision": mean("tool_precision"), "tool_recall": mean("tool_recall"), "tool_f1": mean("tool_f1"),
        "tool_success_rate": mean("tool_success_rate"), "reason_coverage": mean("reason_coverage"), "rubric": mean("rubric"),
        "hallucinated_numbers_total": sum(r["hallucinated_numbers"] for r in rows),
        "fact_mismatch_total": sum(r["fact_mismatch"] for r in rows), "pii_leaks": sum(r["pii_leak"] for r in rows),
        "retrieval_hit_rate": round(statistics.mean(r["hit"] for r in rrows), 4),
        "retrieval_mrr": round(statistics.mean(r["mrr"] for r in rrows), 4),
        "retrieval_precision@3": round(statistics.mean(r["precision@3"] for r in rrows), 4),
        "retrieval_recall@3": round(statistics.mean(r["recall@3"] for r in rrows), 4),
        "retrieval_ndcg@3": round(statistics.mean(r["ndcg@3"] for r in rrows), 4),
        "latency_p50_ms": percentile(lat, 50), "latency_p95_ms": percentile(lat, 95),
        "llm_calls_total": sum(r["llm_calls"] for r in rows), "tokens_total": sum(r["tokens"] for r in rows),
        "cost_usd_total": round(sum(r["cost_usd"] for r in rows), 6),
    }


def gate(summary: dict) -> list[str]:
    failures = []
    for k, thr in THRESHOLDS.items():
        v = summary[k]
        bad = v > thr if k.endswith(("_total", "leaks")) else v < thr
        if bad:
            failures.append(f"{k}={v} (threshold {thr})")
    return failures


def write_md(summary: dict, rows: list[dict], rrows: list[dict], failures: list[str], path: Path) -> None:
    lines = ["# LoanFlow evaluation report", "", f"Gate: {'PASS' if not failures else 'FAIL'}", ""]
    lines += ["| metric | value |", "|---|---|", *[f"| {k} | {v} |" for k, v in summary.items()], ""]
    lines += ["## Cases", "", "| id | decision (exp) | trajectory exact | tool F1 | HITL ok | latency ms |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['decision']} ({r['expected_decision']}) | {r['trajectory_exact']} | {r['tool_f1']:.2f} | "
              f"{r['hitl_correct']} | {r['latency_ms']} |" for r in rows]
    lines += ["", "## Retrieval", "", "| id | query | MRR | P@3 |", "|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['query']} | {r['mrr']:.2f} | {r['precision@3']:.2f} |" for r in rrows]
    if failures:
        lines += ["", "## Gate failures", *[f"- {f}" for f in failures]]
    path.write_text("\n".join(lines))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "golden_dataset.json"))
    ap.add_argument("--out", default=str(HERE / "reports"))
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge scores (needs a live LLM)")
    args = ap.parse_args()
    data = json.loads(Path(args.dataset).read_text())
    await seed_long_term_memory()
    orch = LoanOrchestrator(build_graph(checkpointer=MemorySaver()))
    rows = [await run_case(orch, c, args.judge) for c in data["cases"]]
    rrows = await run_retrieval(data["retrieval"])
    summary = aggregate(rows, rrows)
    failures = gate(summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({"summary": summary, "gate_failures": failures, "cases": rows,
                                                 "retrieval": rrows}, indent=2, default=str))
    write_md(summary, rows, rrows, failures, out / "report.md")
    print(json.dumps(summary, indent=2))
    print("GATE:", "PASS" if not failures else f"FAIL {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
