"""Checkpointer factory.

Postgres (AsyncPostgresSaver) in staging/prod: the graph state is persisted after every
node, so a run can
  * pause for a human (interrupt) for hours/days and resume on another instance,
  * survive a Cloud Run instance being recycled mid-run,
  * be replayed / time-travelled for audit ("what did the supervisor know at step 4?").
MemorySaver for local dev and tests.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver

from loanflow.common.config import get_settings
from loanflow.common.logging import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def checkpointer_ctx():
    dsn = get_settings().postgres_dsn
    if dsn:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
                await saver.setup()
                yield saver
                return
        except ImportError:
            log.warning("langgraph-checkpoint-postgres not installed; falling back to MemorySaver")
    yield MemorySaver()
