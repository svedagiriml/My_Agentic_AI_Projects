"""Bulkheads and deadlines.

* Bulkhead  — bounded concurrency per dependency so one slow sub-agent cannot exhaust the
  supervisor's connection pool / event loop (isolation of failure domains).
* Deadline  — an absolute request deadline propagated through every handoff; each hop only
  spends the *remaining* budget, so a request never outlives the client's patience.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

from bankcore.errors import DependencyError, DependencyTimeout

T = TypeVar("T")
_bulkheads: dict[str, asyncio.Semaphore] = {}


@asynccontextmanager
async def bulkhead(name: str, max_concurrency: int = 20, acquire_timeout_s: float = 2.0):
    sem = _bulkheads.setdefault(name, asyncio.Semaphore(max_concurrency))
    try:
        await asyncio.wait_for(sem.acquire(), timeout=acquire_timeout_s)
    except asyncio.TimeoutError as e:
        raise DependencyError(f"bulkhead '{name}' saturated", user_message="We're experiencing high demand. "
                                                                         "Please try again shortly.") from e
    try:
        yield
    finally:
        sem.release()


def remaining(deadline_ts: float, floor_s: float = 0.5) -> float:
    left = deadline_ts - time.time()
    if left <= floor_s:
        raise DependencyTimeout("request deadline exhausted")
    return left


async def with_timeout(fn: Callable[[], Awaitable[T]], timeout_s: float, what: str) -> T:
    try:
        return await asyncio.wait_for(fn(), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise DependencyTimeout(f"{what} timed out after {timeout_s:.1f}s") from e
