"""End-to-end over real HTTP: 5 MCP server processes + 7 agent service processes (Cloud Run topology)."""
import json
import sys
from pathlib import Path

import httpx
import pytest
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import run_local_stack  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.http]


@pytest.fixture(scope="module")
def stack():
    procs = run_local_stack.start(base_mcp=19101, base_agent=18101, supervisor_port=None)
    yield run_local_stack.stack_env(19101, 18101)
    run_local_stack.stop(procs)


@pytest.fixture
def http_mode(stack, monkeypatch):
    from loanflow.common.config import reset_settings_cache

    for k, v in stack.items():
        monkeypatch.setenv(k, v)
    reset_settings_cache()


async def test_full_run_over_http(http_mode):
    from loanflow.supervisor.graph import build_graph
    from loanflow.supervisor.runner import LoanOrchestrator
    from loanflow.supervisor.transport import HttpTransport

    orch = LoanOrchestrator(build_graph(checkpointer=MemorySaver(), transport=HttpTransport()))
    st = await orch.execute("run_http_1", "A1001", "D-1001-APP", "officer")
    assert st.status == "completed" and st.decision.decision.value == "APPROVED"
    assert st.trajectory[-1] == "compile_report"


async def test_agent_rejects_unauthenticated_and_unsigned(http_mode):
    url = json.loads(run_local_stack.stack_env(19101, 18101)["AGENT_URLS"])["check_credit"]
    async with httpx.AsyncClient() as c:
        assert (await c.post(f"{url}/v1/tasks", json={})).status_code == 401
        r = await c.post(f"{url}/v1/tasks", headers={"Authorization": "Bearer dev-internal-token"},
                         json={"run_id": "r", "recipient": "check_credit", "applicant_id": "A1001",
                               "document_id": "D-1001-APP", "instruction": "x"})
        assert r.status_code == 400 and r.json()["error"]["code"] == "MESSAGE_INTEGRITY_FAILED"


async def test_mcp_server_requires_service_token(http_mode):
    async with httpx.AsyncClient() as c:
        r = await c.post("http://127.0.0.1:19101/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert r.status_code == 401
