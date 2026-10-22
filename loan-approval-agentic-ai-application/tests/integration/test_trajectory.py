"""Trajectory-level integration tests over the full multi-agent graph (golden dataset)."""
import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from loanflow.common.exceptions import UpstreamUnavailable
from loanflow.common.schemas import AgentResult, AgentTask
from loanflow.memory.long_term import get_long_term_memory
from loanflow.memory.short_term import ShortTermMemory
from loanflow.supervisor.graph import build_graph
from loanflow.supervisor.runner import LoanOrchestrator
from loanflow.supervisor.transport import InProcessTransport

pytestmark = pytest.mark.integration
GOLDEN = json.loads((Path(__file__).parents[2] / "eval" / "golden_dataset.json").read_text())["cases"]


@pytest.mark.parametrize("case", GOLDEN, ids=[c["id"] for c in GOLDEN])
async def test_golden_trajectory(orchestrator, case):
    run_id = f"run_it_{case['id'].lower()}"
    st = await orchestrator.execute(run_id, case["applicant_id"], case["document_id"], "officer")
    exp = case["expected"]
    assert (st.status == "awaiting_human") == exp["hitl"]
    if exp["hitl"]:
        assert st.pending_review and st.pending_review["reasons"]
        st = await orchestrator.resume(run_id, case["human_review"], "underwriter-7")
    assert st.status == exp["status"]
    assert st.trajectory == exp["trajectory"]
    if exp["decision"]:
        assert st.decision.decision.value == exp["decision"]
        assert st.decision.audit_trail and st.decision.report_sha256
        for k, v in exp["facts"].items():
            assert getattr(st.decision, k) == v
    else:
        assert st.error.code.endswith(exp["error_code"]) or st.error.code == exp["error_code"]
    audit = await orchestrator.audit(run_id)
    # every tool call came from the expected tool set of its node
    for call in audit["tool_calls"]:
        assert call["tool"] in exp["tools"][call["node"]]
    # Chain-of-Thought routing log is recorded for every supervisor step
    assert all(r["thoughts"] for r in audit["routing_log"])


async def test_checkpoint_resume_on_another_instance():
    """HITL pause -> process 'dies' -> a new orchestrator sharing the checkpointer resumes the run."""
    saver = MemorySaver()  # stands in for AsyncPostgresSaver shared by all Cloud Run instances
    a = LoanOrchestrator(build_graph(checkpointer=saver))
    st = await a.execute("run_ckpt", "A1005", "D-1005-APP", "officer")
    assert st.status == "awaiting_human"
    b = LoanOrchestrator(build_graph(checkpointer=saver))  # fresh graph, same durable state
    assert (await b.status("run_ckpt")).status == "awaiting_human"
    st = await b.resume("run_ckpt", {"decision": "APPROVED", "reviewer_notes": "verified salary slips"}, "uw-2")
    assert st.decision.decision.value == "APPROVED" and st.decision.human_review.reviewer == "uw-2"
    assert st.trajectory.count("check_credit") == 1  # no node re-executed after resume


async def test_invalid_human_review_rejected(orchestrator):
    from loanflow.common.exceptions import ValidationFailure

    await orchestrator.execute("run_bad_review", "A1001", "D-1001-APP", "officer")
    with pytest.raises(ValidationFailure):  # run is complete, not awaiting review
        await orchestrator.resume("run_bad_review", {"decision": "APPROVED", "reviewer_notes": "xxxxx"}, "uw")


class FlakyTransport(InProcessTransport):
    def __init__(self, fail_node: str, failures: int):
        self.fail_node, self.failures = fail_node, failures

    async def invoke(self, task: AgentTask) -> AgentResult:
        if task.recipient == self.fail_node and self.failures > 0:
            self.failures -= 1
            raise UpstreamUnavailable(f"{task.recipient} instance cold / 503")
        return await super().invoke(task)


async def test_transient_agent_failure_is_retried():
    orch = LoanOrchestrator(build_graph(checkpointer=MemorySaver(), transport=FlakyTransport("check_credit", 1)))
    st = await orch.execute("run_flaky", "A1001", "D-1001-APP", "officer")
    assert st.decision.decision.value == "APPROVED"
    assert st.trajectory.count("check_credit") == 2
    assert any("retrying" in n for n in await ShortTermMemory("run_flaky").notes())


async def test_persistent_failure_fails_safely():
    orch = LoanOrchestrator(build_graph(checkpointer=MemorySaver(), transport=FlakyTransport("check_credit", 99)))
    st = await orch.execute("run_down", "A1001", "D-1001-APP", "officer")
    assert st.status == "failed" and st.decision is None and st.error.code == "UPSTREAM_UNAVAILABLE"


async def test_outcome_written_to_episodic_memory(orchestrator):
    await orchestrator.execute("run_mem", "A1003", "D-1003-APP", "officer")
    hits = await get_long_term_memory().recall("previous loan decision rejected application", kinds=["episodic"], namespaces=["applicant:A1003"])
    assert hits and "REJECTED" in hits[0].record.content and hits[0].record.source == "compile_rejection"


async def test_conversation_history_in_short_term_memory(orchestrator):
    await orchestrator.execute("run_hist", "A1001", "D-1001-APP", "officer")
    hist = await ShortTermMemory("run_hist").history(last_n=50)
    roles = [h["role"] for h in hist]
    assert roles[0] == "supervisor" and "compile_report" in roles
