"""Human-in-the-loop evaluation of end results.

Step 1  python -m eval.run_eval                       -> eval/reports/report.json
Step 2  python -m eval.hitl_review label              -> a credit expert reviews each final decision
        (interactive: correct? 1-5 quality rating, comment) -> eval/reports/human_labels.jsonl
        or: python -m eval.hitl_review label --from-csv labels.csv   (bulk import from a review sheet)
Step 3  python -m eval.hitl_review summarize          -> human agreement metrics:
        human-judged accuracy, mean quality rating, system-vs-human Cohen's kappa,
        list of disagreements (feed these back into the golden dataset).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from eval.metrics import cohen_kappa

REPORTS = Path(__file__).parent / "reports"
LABELS = REPORTS / "human_labels.jsonl"


def _cases() -> list[dict]:
    return json.loads((REPORTS / "report.json").read_text())["cases"]


def label(from_csv: str | None, reviewer: str) -> None:
    done = {json.loads(line)["id"] for line in LABELS.read_text().splitlines()} if LABELS.exists() else set()
    rows: list[dict] = []
    if from_csv:
        with open(from_csv) as f:
            for r in csv.DictReader(f):
                rows.append({"id": r["id"], "human_decision": r["human_decision"].upper(), "quality": int(r["quality"]),
                             "comment": r.get("comment", ""), "reviewer": r.get("reviewer", reviewer)})
    else:
        for c in _cases():
            if c["id"] in done:
                continue
            print("\n" + "=" * 80)
            print(f"{c['id']}  system decision: {c['decision']}   trajectory: {' > '.join(c['trajectory'])}")
            print(f"summary: {c.get('final_summary')}")
            hd = input("Your decision [APPROVED/REJECTED/NONE, enter=agree]: ").strip().upper() or str(c["decision"]).upper()
            q = int(input("Quality of explanation 1-5 [4]: ").strip() or 4)
            rows.append({"id": c["id"], "human_decision": hd, "quality": q, "comment": input("Comment: ").strip(),
                         "reviewer": reviewer})
    REPORTS.mkdir(exist_ok=True)
    with LABELS.open("a") as f:
        for r in rows:
            f.write(json.dumps({**r, "labelled_at": datetime.now(UTC).isoformat()}) + "\n")
    print(f"saved {len(rows)} labels -> {LABELS}")


def summarize() -> dict:
    labels = {(r := json.loads(line))["id"]: r for line in LABELS.read_text().splitlines()}
    cases = [c for c in _cases() if c["id"] in labels]
    sys_d = [str(c["decision"]).upper() for c in cases]
    hum_d = [labels[c["id"]]["human_decision"] for c in cases]
    out = {
        "labelled": len(cases),
        "human_agreement_accuracy": round(sum(a == b for a, b in zip(sys_d, hum_d)) / max(len(cases), 1), 4),
        "cohen_kappa": round(cohen_kappa(sys_d, hum_d), 4),
        "mean_quality_rating": round(statistics.mean(labels[c["id"]]["quality"] for c in cases), 2) if cases else 0,
        "disagreements": [{"id": c["id"], "system": s, "human": h, "comment": labels[c["id"]].get("comment")}
                          for c, s, h in zip(cases, sys_d, hum_d) if s != h],
    }
    (REPORTS / "human_eval_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("label")
    lp.add_argument("--from-csv")
    lp.add_argument("--reviewer", default="credit-expert")
    sub.add_parser("summarize")
    a = ap.parse_args()
    label(a.from_csv, a.reviewer) if a.cmd == "label" else summarize()
