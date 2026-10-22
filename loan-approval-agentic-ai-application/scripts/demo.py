"""Demo client: submits the sample applications, polls, and plays the underwriter for HITL.

  make run-local            # terminal 1
  python scripts/demo.py    # terminal 2   (BASE_URL=http://localhost:8000)
"""
from __future__ import annotations

import json
import os
import time

import httpx

BASE = os.environ.get("BASE_URL", "http://localhost:8000")
CASES = [("A1001", "D-1001-APP"), ("A1002", "D-1002-APP"), ("A1003", "D-1003-APP"), ("A1004", "D-1004-APP"),
         ("A1005", "D-1005-APP"), ("A1006", "D-1006-APP"), ("A1007", "D-1007-APP"), ("A1008", "D-1008-APP")]
HUMAN = {"A1005": ("APPROVED", "Income verified via Form 16"), "A1006": ("REJECTED", "Tampered document - fraud referral"),
         "A1008": ("REJECTED", "Volatile business turnover")}


def token(c: httpx.Client, sub: str, roles: list[str]) -> dict[str, str]:
    t = c.post(f"{BASE}/v1/dev/token", json={"sub": sub, "roles": roles}).json()["access_token"]
    return {"Authorization": f"Bearer {t}"}


def main() -> None:
    with httpx.Client(timeout=120) as c:
        officer, uw = token(c, "officer-demo", ["loan_officer"]), token(c, "uw-demo", ["underwriter"])
        runs = {}
        for aid, did in CASES:
            r = c.post(f"{BASE}/v1/loan-applications", json={"applicant_id": aid, "document_id": did}, headers=officer)
            r.raise_for_status()
            runs[aid] = r.json()["run_id"]
            print(f"submitted {aid} -> {runs[aid]} (202 queued)")
        pending = set(runs)
        while pending:
            for aid in sorted(pending):
                st = c.get(f"{BASE}/v1/runs/{runs[aid]}", headers=uw).json()
                if st["status"] == "awaiting_human":
                    dec, notes = HUMAN.get(aid, ("REJECTED", "default reviewer decision"))
                    print(f"  HITL {aid}: {st['pending_review']['reasons']} -> underwriter {dec}")
                    c.post(f"{BASE}/v1/runs/{runs[aid]}/review", json={"decision": dec, "reviewer_notes": notes}, headers=uw)
                elif st["status"] in ("completed", "failed"):
                    d = st.get("decision") or {}
                    print(f"{aid}: {st['status']:<9} {d.get('decision', '-'):<9} {' > '.join(st['trajectory'])}")
                    if d:
                        print(f"      reasons: {json.dumps(d['reasons'])[:160]}")
                    pending.discard(aid)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
