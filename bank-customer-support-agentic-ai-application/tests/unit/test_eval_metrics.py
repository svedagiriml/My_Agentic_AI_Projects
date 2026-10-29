"""Evaluation metric correctness (the evaluators need tests too)."""
from __future__ import annotations

import pytest

from eval import metrics as M


def test_trajectory_scores():
    exp = ["account:get_account_balance", "cards:report_lost_stolen_card"]
    assert M.trajectory_scores(exp, exp)["trajectory_exact"] == 1.0
    rev = M.trajectory_scores(exp, list(reversed(exp)))
    assert rev["trajectory_exact"] == 0 and rev["trajectory_in_order"] == 0 and rev["trajectory_any_order"] == 1
    extra = M.trajectory_scores(exp, [exp[0], "cards:order_replacement_card", exp[1]])
    assert extra["trajectory_in_order"] == 1 and extra["trajectory_exact"] == 0


def test_tool_prf():
    assert M.set_prf(["a", "b"], ["a"]) == (1.0, 0.5, pytest.approx(2 / 3))
    assert M.set_prf([], []) == (1.0, 1.0, 1.0)


def test_retrieval_scores_bounded():
    s = M.retrieval_scores(["fees"], ["cards", "fees", "hours"], k=4)
    assert s["recall@4"] == 1 and s["mrr"] == 0.5 and 0 < s["ndcg@4"] <= 1


def test_fact_recall_and_hallucination():
    assert M.fact_recall("Balance is $5,240.75", ["5240.75"]) == 1.0
    assert M.hallucination_rate("Balance is $99.00", ['{"b": "5240.75"}']) == 1.0
    assert M.hallucination_rate("Balance is $5240.75", ['{"b": "5240.75"}']) == 0.0


def test_cohen_kappa():
    assert M.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert M.cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
