"""Message/data validation: every handoff contract rejects malformed or inconsistent data."""
from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from bankcore.schemas.common import AgentName, RiskLevel
from bankcore.schemas.messages import (
    ChatRequest, PlannedTask, ProposedAction, SubAgentRequest, SubAgentResponse, SupervisorPlan,
)
from bankcore.schemas.tools import BillPayInput, DateRange, InternalTransferInput, TOOL_INPUT_MODELS, TOOL_OUTPUT_MODELS


def _task(tid, agent="account", deps=None):
    return {"task_id": tid, "agent": agent, "instruction": "do something useful", "depends_on": deps or []}


def test_valid_multi_intent_plan():
    p = SupervisorPlan(reasoning=["a"], tasks=[_task("t1"), _task("t2", "cards", ["t1"])], confidence=0.9)
    assert [t.agent for t in p.tasks] == [AgentName.ACCOUNT, AgentName.CARDS]


@pytest.mark.parametrize("tasks,err", [
    ([_task("t1"), _task("t1", "cards")], "duplicate"),
    ([_task("t1", deps=["t9"])], "unknown"),
    ([_task("t1", deps=["t2"]), _task("t2", deps=["t1"])], "cycle"),
    ([_task("t1", "compliance")], "compliance"),
    ([], "tasks or ask"),
])
def test_invalid_plans_rejected(tasks, err):
    with pytest.raises(ValidationError, match=err):
        SupervisorPlan(reasoning=["x"], tasks=tasks)


def test_plan_rejects_unknown_fields_and_agents():
    with pytest.raises(ValidationError):
        SupervisorPlan(reasoning=["x"], tasks=[_task("t1", "hacker")])
    with pytest.raises(ValidationError):
        PlannedTask(**_task("t1"), sneaky="field")


def test_chat_request_limits():
    with pytest.raises(ValidationError):
        ChatRequest(message="")
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 4001)
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", session_id="../../etc")


def test_proposed_action_hash_detects_tampering():
    a = ProposedAction(agent="payments", tool="initiate_internal_transfer", risk_level=RiskLevel.HIGH,
                       arguments={"amount": "1500.00"}, reason="r", summary_for_reviewer="s")
    data = a.model_dump()
    data["arguments"]["amount"] = "9500.00"           # tamper after hashing
    with pytest.raises(ValidationError, match="tampered"):
        ProposedAction.model_validate(data)


def test_subagent_response_consistency():
    with pytest.raises(ValidationError):
        SubAgentResponse(task_id="t1", agent="payments", status="needs_approval")
    with pytest.raises(ValidationError):
        SubAgentResponse(task_id="t1", agent="payments", status="failed")


def test_execute_approved_requires_token():
    with pytest.raises(ValidationError):
        SubAgentRequest(task_id="t1", thread_id="th", user_id="USR-1001", instruction="x", mode="execute_approved",
                        delegation_token="d")


def test_money_canonicalisation_and_limits():
    a = InternalTransferInput(source_acc="ACC-100001", dest_acc="ACC-100002", amount=250)
    b = InternalTransferInput(source_acc="ACC-100001", dest_acc="ACC-100002", amount="250.0")
    assert a.model_dump(mode="json") == b.model_dump(mode="json")
    assert a.amount == Decimal("250.00")
    for bad in (0, -5, "1000000.01", "10.123"):
        with pytest.raises(ValidationError):
            InternalTransferInput(source_acc="ACC-100001", dest_acc="ACC-100002", amount=bad)
    with pytest.raises(ValidationError, match="differ"):
        InternalTransferInput(source_acc="ACC-100001", dest_acc="ACC-100001", amount=5)


def test_id_formats_and_dates():
    with pytest.raises(ValidationError):
        InternalTransferInput(source_acc="12345678", dest_acc="ACC-100002", amount=5)
    with pytest.raises(ValidationError):
        DateRange(start="2026-02-01", end="2026-01-01")
    with pytest.raises(ValidationError):
        DateRange(start="2024-01-01", end="2026-01-01")
    with pytest.raises(ValidationError, match="past"):
        BillPayInput(payee_id="PAY-5001", amount=10, date="2020-01-01")


def test_every_tool_has_input_and_output_contract():
    assert set(TOOL_INPUT_MODELS) == set(TOOL_OUTPUT_MODELS)
    assert len(TOOL_OUTPUT_MODELS) == 25
