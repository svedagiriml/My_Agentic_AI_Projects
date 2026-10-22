"""Real Redis backend: KV ops, rate limiting, semantic cache, Streams job queue, full run.

Runs only when REDIS_TEST_URL is set, e.g.
  redis-server --port 6390 &  REDIS_TEST_URL=redis://localhost:6390/15 pytest tests/integration/test_redis_backend.py
"""
import asyncio
import os

import pytest
from langgraph.checkpoint.memory import MemorySaver

from loanflow.memory import kv as kv_mod

URL = os.environ.get("REDIS_TEST_URL")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not URL, reason="REDIS_TEST_URL not set")]


@pytest.fixture
async def redis_kv(monkeypatch):
    from loanflow.common.config import reset_settings_cache

    monkeypatch.setenv("REDIS_URL", URL)
    reset_settings_cache()
    kv = kv_mod.RedisKV(URL)
    await kv.raw.flushdb()
    kv_mod.set_kv(kv)
    yield kv
    await kv.raw.flushdb()
    await kv.raw.aclose()


async def test_kv_and_rate_limit(redis_kv):
    from loanflow.common.exceptions import RateLimitExceeded
    from loanflow.security.rate_limit import check_rate_limit

    await redis_kv.set("a", {"x": 1}, ttl_s=10)
    assert await redis_kv.get("a") == {"x": 1}
    await redis_kv.rpush("l", 1, maxlen=2)
    await redis_kv.rpush("l", 2, maxlen=2)
    await redis_kv.rpush("l", 3, maxlen=2)
    assert await redis_kv.lrange("l") == [2, 3]
    for _ in range(2):
        await check_rate_limit("u", limit_per_minute=2)
    with pytest.raises(RateLimitExceeded):
        await check_rate_limit("u", limit_per_minute=2)


async def test_semantic_cache_on_redis(redis_kv):
    from loanflow.memory.semantic_cache import SemanticCache

    c = SemanticCache("routing")
    await c.put("done=fetch_document | doc_valid=na", {"next_node": "validate_document"})
    assert (await c.get("done=fetch_document | doc_valid=na"))[0]["next_node"] == "validate_document"


async def test_stream_queue_runs_jobs_and_survives_failures(redis_kv):
    from loanflow.supervisor.graph import build_graph
    from loanflow.supervisor.queue import DLQ, RedisStreamQueue
    from loanflow.supervisor.runner import LoanOrchestrator

    orch = LoanOrchestrator(build_graph(checkpointer=MemorySaver()))
    done: list[str] = []

    async def handler(job):
        if job["run_id"] == "run_poison":
            raise RuntimeError("poison message")
        st = await orch.execute(job["run_id"], job["applicant_id"], job["document_id"], "officer")
        done.append(st.decision.decision.value)

    q = RedisStreamQueue(redis_kv, claim_idle_ms=200, block_ms=100)
    await q.start(handler, concurrency=4)
    for i, (a, d) in enumerate([("A1001", "D-1001-APP"), ("A1003", "D-1003-APP")]):
        await q.enqueue({"type": "start", "run_id": f"run_q{i}", "applicant_id": a, "document_id": d})
    await q.enqueue({"type": "start", "run_id": "run_poison", "applicant_id": "A1001", "document_id": "D-1001-APP"})
    for _ in range(100):
        if len(done) == 2 and await redis_kv.raw.xlen(DLQ) == 1:
            break
        await asyncio.sleep(0.1)
    await q.stop()
    assert sorted(done) == ["APPROVED", "REJECTED"]
    assert (await redis_kv.get("stm:run_q0:status"))["status"] == "completed"  # short-term memory in Redis
    # poison job: redelivered via XAUTOCLAIM until MAX_DELIVERIES, then dead-lettered and ACKed
    assert await redis_kv.raw.xlen(DLQ) == 1
    assert (await redis_kv.raw.xpending("loan:jobs", "supervisors"))["pending"] == 0
