"""Registry of workflows paused for human approval (staff console queue).

The authoritative paused state lives in the LangGraph checkpoint (Postgres); this registry is
just a fast index for the staff UI: thread_id -> who/what/when. SLA: approvals expire after
``ttl_s``; expired requests are auto-rejected by the console job (fail-safe default = no money moves).
"""
from __future__ import annotations

import json
import time
from typing import Any


class HitlRegistry:
    KEY = "hitl:pending"

    def __init__(self, redis=None, ttl_s: int = 3600) -> None:
        self.r, self.ttl = redis, ttl_s
        self._mem: dict[str, dict] = {}

    async def add(self, thread_id: str, item: dict[str, Any]) -> None:
        item = {**item, "thread_id": thread_id, "created_at": time.time(), "expires_at": time.time() + self.ttl}
        if self.r is not None:
            try:
                await self.r.hset(self.KEY, thread_id, json.dumps(item, default=str))
                return
            except Exception:  # noqa: BLE001
                pass
        self._mem[thread_id] = item

    async def get(self, thread_id: str) -> dict | None:
        if self.r is not None:
            try:
                raw = await self.r.hget(self.KEY, thread_id)
                if raw:
                    return json.loads(raw)
            except Exception:  # noqa: BLE001
                pass
        return self._mem.get(thread_id)

    async def remove(self, thread_id: str) -> None:
        if self.r is not None:
            try:
                await self.r.hdel(self.KEY, thread_id)
            except Exception:  # noqa: BLE001
                pass
        self._mem.pop(thread_id, None)

    async def list(self) -> list[dict]:
        items: list[dict] = []
        if self.r is not None:
            try:
                items = [json.loads(v) for v in (await self.r.hgetall(self.KEY)).values()]
            except Exception:  # noqa: BLE001
                items = []
        items += [v for k, v in self._mem.items() if k not in {i["thread_id"] for i in items}]
        return sorted(items, key=lambda i: i["created_at"])
