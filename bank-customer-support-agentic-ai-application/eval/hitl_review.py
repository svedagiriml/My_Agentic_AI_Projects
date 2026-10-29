"""Human-in-the-loop evaluation: bank SMEs review sampled outputs; we measure how well the
automatic metrics agree with humans (Cohen's kappa) so automated gates stay trustworthy.

    python eval/hitl_review.py review --report eval/reports/report.json --sample 15 --reviewer STF-001
    python eval/hitl_review.py score  --labels eval/reports/human_labels.jsonl --report eval/reports/report.json

Labels are JSONL: {"case_id", "reviewer_id", "correct": bool, "helpful": 1-5, "safe": bool, "notes"}
(and can be bulk-loaded into the ``hitl_reviews`` table for audit).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import cohen_kappa, mean  # noqa: E402


def _auto_pass(case: dict) -> bool:
    return (case["fact_recall"] >= 1.0 and case["tool_f1"] >= 1.0 and case["hallucination"] == 0
            and case["status"] == case["expected_status"])


def review(report: Path, sample: int, reviewer: str, out: Path) -> None:
    cases = json.loads(report.read_text())["cases"]
    random.Random(7).shuffle(cases)
    with out.open("a") as f:
        for c in cases[:sample]:
            print("=" * 80, f"\nCASE {c['id']} ({c['category']})  status={c['status']}  agents={c['agents']}")
            print(f"steps: {c['steps']}\nREPLY:\n{c['reply']}\n")
            correct = input("Correct & complete? [y/n] ").strip().lower() == "y"
            helpful = int(input("Helpfulness 1-5: ").strip() or 3)
            safe = input("Safe/compliant? [y/n] ").strip().lower() == "y"
            notes = input("Notes (optional): ").strip()
            f.write(json.dumps({"case_id": c["id"], "reviewer_id": reviewer, "correct": correct,
                                "helpful": helpful, "safe": safe, "notes": notes}) + "\n")
    print(f"labels appended to {out}")


def score(labels: Path, report: Path) -> dict:
    cases = {c["id"]: c for c in json.loads(report.read_text())["cases"]}
    rows = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["case_id"] in cases]
    human = [r["correct"] for r in rows]
    auto = [_auto_pass(cases[r["case_id"]]) for r in rows]
    result = {"n_labels": len(rows), "human_accuracy": mean([1.0 if h else 0.0 for h in human]),
              "auto_accuracy": mean([1.0 if a else 0.0 for a in auto]),
              "agreement": mean([1.0 if h == a else 0.0 for h, a in zip(human, auto)]),
              "cohen_kappa": cohen_kappa(human, auto),
              "human_safety_rate": mean([1.0 if r["safe"] else 0.0 for r in rows]),
              "mean_helpfulness": mean([float(r["helpful"]) for r in rows]),
              "disagreements": [r["case_id"] for r, a in zip(rows, auto) if r["correct"] != a]}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("review")
    r.add_argument("--report", default="eval/reports/report.json")
    r.add_argument("--sample", type=int, default=10)
    r.add_argument("--reviewer", required=True)
    r.add_argument("--out", default="eval/reports/human_labels.jsonl")
    s = sub.add_parser("score")
    s.add_argument("--labels", default="eval/reports/human_labels.jsonl")
    s.add_argument("--report", default="eval/reports/report.json")
    a = ap.parse_args()
    if a.cmd == "review":
        review(Path(a.report), a.sample, a.reviewer, Path(a.out))
    else:
        score(Path(a.labels), Path(a.report))
