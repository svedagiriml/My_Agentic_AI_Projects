"""Asynchronous job queue for runs (decouples the API from long-running graph execution).

  API  --XADD-->  Redis Stream `loan:jobs`  --XREADGROUP (consumer group)-->  workers
                                     |-- failures > MAX_DELIVERIES --> `loan:jobs:dlq`

* Horizontal scaling: any number of supervisor-worker instances join the same consumer
  group; each job is delivered to exactly one of them.
* At-least-once + idempotent: jobs carry the run_id (= LangGraph thread_id); re-running a
  job resumes from the last checkpoint instead of duplicating work.
* Crash recovery: XAUTOCLAIM re-delivers jobs that a dead worker never ACKed. A heartbeat
  (XCLAIM JUSTID) keeps in-progress jobs from being stolen, and a per-run Redis lock
  guarantees one run never executes concurrently on two workers.
* GCP alternative: Pub/Sub push subscription -> /internal/jobs (same handler).
"""
from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from loanflow.common.config import get_settings
from loanflow.common.logging import get_logger
from loanflow.memory.kv import RedisKV, get_kv

log = get_logger(__name__)
STREAM, GROUP, DLQ = "loan:jobs", "supervisors", "loan:jobs:dlq"
MAX_DELIVERIES = 3
Handler = Callable[[dict[str, Any]], Awaitable[None]]


class JobQueue(Protocol):
    async def enqueue(self, job: dict[str, Any]) -> str: ...
    async def start(self, handler: Handler, concurrency: int) -> None: ...
    async def stop(self) -> None: ...


class LocalQueue:
    """In-process queue for local dev / tests (same retry + DLQ semantics)."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._tasks: list[asyncio.Task] = []
        self.dlq: list[dict[str, Any]] = []

    async def enqueue(self, job):
        await self._q.put(job)
        return job["run_id"]

    async def start(self, handler, concurrency):
        async def worker():
            while True:
                job = await self._q.get()
                for attempt in range(1, MAX_DELIVERIES + 1):
                    try:
                        await handler(job)
                        break
                    except Exception as exc:  # noqa: BLE001
                        log.warning("job failed", extra={"extra_fields": {"run_id": job.get("run_id"), "attempt": attempt, "error": str(exc)}})
                        if attempt == MAX_DELIVERIES:
                            self.dlq.append({**job, "error": str(exc)})
                        else:
                            await asyncio.sleep(0.5 * attempt)
                self._q.task_done()

        self._tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]

    async def join(self):
        await self._q.join()

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


class RedisStreamQueue:
    def __init__(self, kv: RedisKV, claim_idle_ms: int = 300_000, block_ms: int = 5000):
        self.r = kv.raw
        self.consumer = f"{socket.gethostname()}-{id(self)}"
        self.claim_idle_ms = claim_idle_ms  # re-deliver jobs un-ACKed for this long (dead worker / failed job)
        self.block_ms = block_ms
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._inflight: set[str] = set()

    async def enqueue(self, job):
        await self.r.xadd(STREAM, {"job": json.dumps(job)}, maxlen=100_000, approximate=True)
        return job["run_id"]

    async def _ensure_group(self):
        try:
            await self.r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except Exception as exc:  # noqa: BLE001 - BUSYGROUP is fine
            if "BUSYGROUP" not in str(exc):
                raise

    async def _heartbeat(self, msg_id: str, lock_key: str) -> None:
        """Reset the job's idle time and renew the run lock while it is being processed."""
        interval = max(self.claim_idle_ms / 3000, 0.05)
        while True:
            await asyncio.sleep(interval)
            await self.r.xclaim(STREAM, GROUP, self.consumer, min_idle_time=0, message_ids=[msg_id], justid=True)
            await self.r.pexpire(lock_key, self.claim_idle_ms * 2)

    async def _handle(self, handler, msg_id, fields):
        job = json.loads(fields["job"])
        lock_key = f"lock:run:{job.get('run_id')}"
        # distributed per-run lock: the same run never executes concurrently on two workers
        if not await self.r.set(lock_key, self.consumer, nx=True, px=self.claim_idle_ms * 2):
            self._inflight.discard(msg_id)
            return  # another worker owns it; left un-ACKed and reclaimed if that worker dies
        hb = asyncio.create_task(self._heartbeat(msg_id, lock_key))
        try:
            await handler(job)
            await self.r.xack(STREAM, GROUP, msg_id)
        except Exception as exc:  # noqa: BLE001
            info = await self.r.xpending_range(STREAM, GROUP, min=msg_id, max=msg_id, count=1)
            deliveries = info[0]["times_delivered"] if info else MAX_DELIVERIES
            log.warning("job failed", extra={"extra_fields": {"run_id": job.get("run_id"), "deliveries": deliveries, "error": str(exc)}})
            if deliveries >= MAX_DELIVERIES:
                # idempotent dead-lettering: exactly one DLQ entry per message even under redelivery races
                if await self.r.set(f"dlq:done:{msg_id}", "1", nx=True, ex=7 * 86_400):
                    await self.r.xadd(DLQ, {"job": fields["job"], "error": str(exc)[:500], "msg_id": msg_id})
                await self.r.xack(STREAM, GROUP, msg_id)
        finally:
            hb.cancel()
            self._inflight.discard(msg_id)
            if await self.r.get(lock_key) == self.consumer:
                await self.r.delete(lock_key)

    async def start(self, handler, concurrency):
        await self._ensure_group()
        self._running = True
        sem = asyncio.Semaphore(concurrency)

        async def loop():
            while self._running:
                try:
                    claimed = await self.r.xautoclaim(STREAM, GROUP, self.consumer, min_idle_time=self.claim_idle_ms, count=5)
                    batches = [(STREAM, claimed[1])] if claimed and claimed[1] else []
                    batches += await self.r.xreadgroup(GROUP, self.consumer, {STREAM: ">"}, count=concurrency,
                                                       block=self.block_ms) or []
                    for _, msgs in batches:
                        for msg_id, fields in msgs:
                            if msg_id in self._inflight or fields is None:
                                continue
                            self._inflight.add(msg_id)  # mark before scheduling: closes the re-dispatch window
                            await sem.acquire()
                            t = asyncio.create_task(self._handle(handler, msg_id, fields))
                            t.add_done_callback(lambda _: sem.release())
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    log.error("queue loop error", extra={"extra_fields": {"error": str(exc)}})
                    await asyncio.sleep(2)

        self._tasks = [asyncio.create_task(loop())]

    async def stop(self):
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


def make_queue() -> JobQueue:
    kv = get_kv()
    return RedisStreamQueue(kv) if isinstance(kv, RedisKV) and get_settings().redis_url else LocalQueue()
