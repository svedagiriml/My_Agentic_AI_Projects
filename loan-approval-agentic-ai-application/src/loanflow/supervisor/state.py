"""LangGraph state for one loan run (checkpointed after every super-step)."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    return {**(a or {}), **(b or {})}


class LoanState(TypedDict, total=False):
    run_id: str
    applicant_id: str
    document_id: str
    officer: str
    notes: list[str]                                         # short-term notes (also mirrored to Redis)
    results: Annotated[dict[str, Any], merge_dicts]          # node -> validated output
    attempts: Annotated[dict[str, int], merge_dicts]         # node -> attempts so far
    trajectory: Annotated[list[str], operator.add]           # executed nodes, in order
    routing_log: Annotated[list[dict[str, Any]], operator.add]  # supervisor CoT decisions
    audit: Annotated[list[dict[str, Any]], operator.add]     # audit entries
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    metrics: Annotated[dict[str, float], merge_dicts]
    next_node: str
    instruction: str
    error: dict[str, Any] | None
    fatal: bool
    review_reasons: list[str]
    human_review: dict[str, Any] | None
    final_decision: dict[str, Any] | None
    steps: int
