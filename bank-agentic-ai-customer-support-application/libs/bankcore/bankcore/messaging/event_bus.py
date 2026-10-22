"""Asynchronous event bus (fire-and-forget work off the request path).

Topics:
  memory.consolidate  -> memory worker writes episodic/semantic/procedural memories
  audit.events        -> immutable audit trail (compliance, SOX, RBI audit requirements)
  hitl.requested      -> notify staff console / Slack / e-mail that an approval is pending
  hitl.decided        -> analytics + memory (rejected actions are never learned as procedures)

Backends:
  RedisStreamsBus  — local / GKE: consumer groups, at-least-once delivery, XAUTOCLAIM for
                     crashed consumers, dead-letter stream after N delivery attempts.
  PubSubBus        — Cloud Run: Google Pub/Sub topics with push subscriptions to the memory
                     worker service (retry policy + dead-letter topic configured in Terraform/gcloud).
  InMemoryBus      — tests.
Consumers MUST be idempotent (events carry event_id; handlers dedupe).
"""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from bankcore.logging import get_logger

log = get_logger(__name__)
Handler = Callable[[dict[str, Any]], Awaitable[None]]

TOPIC_MEMORY = "memory.consolidate"
TOPIC_AUDIT = "audit.events"
TOPIC_HITL_REQUESTED = "hitl.requested"
TOPIC_HITL_DECIDED = "hitl.decided"


class EventBus(ABC):
    @abstractmethod
    async def publish(self, topic: str, event: dict[str, Any]) -> None: ...

    async def safe_publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publishing must never fail the customer request; failures are logged & metered."""
        try:
            await self.publish(topic, event)
        except Exception as e:  # noqa: BLE001
            log.error("event publish failed", topic=topic, error=str(e))


class InMemoryBus(EventBus):
    def __init__(self) -> None:
        self.events: dict[str, list[dict]] = defaultdict(list)
        self.handlers: dict[str, list[Handler]] = defaultdict(list)

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        self.events[topic].append(event)
        for h in self.handlers[topic]:
            await h(event)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self.handlers[topic].append(handler)


class RedisStreamsBus(EventBus):
    def __init__(self, redis, *, maxlen: int = 100_000, max_deliveries: int = 5) -> None:
        self.r, self.maxlen, self.max_deliveries = redis, maxlen, max_deliveries

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        await self.r.xadd(f"stream:{topic}", {"data": json.dumps(event, default=str)}, maxlen=self.maxlen,
                          approximate=True)

    async def consume(self, topic: str, group: str, consumer: str, handler: Handler, *,
                      stop: asyncio.Event | None = None, block_ms: int = 5000) -> None:
        stream = f"stream:{topic}"
        try:
            await self.r.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception as e:  # noqa: BLE001
            if "BUSYGROUP" not in str(e):
                raise
        stop = stop or asyncio.Event()
        while not stop.is_set():
            # 1) reclaim messages stuck with dead consumers for > 60s
            try:
                claimed = await self.r.xautoclaim(stream, group, consumer, min_idle_time=60_000, count=10)
                batch = [(stream.encode(), claimed[1])] if claimed and claimed[1] else []
            except Exception:  # noqa: BLE001
                batch = []
            if not batch:
                batch = await self.r.xreadgroup(group, consumer, {stream: ">"}, count=20, block=block_ms) or []
            for _stream, messages in batch:
                for msg_id, fields in messages:
                    await self._handle(stream, group, msg_id, fields, handler)

    async def _handle(self, stream: str, group: str, msg_id, fields: dict, handler: Handler) -> None:
        raw = fields.get(b"data") or fields.get("data")
        try:
            await handler(json.loads(raw))
            await self.r.xack(stream, group, msg_id)
        except Exception as e:  # noqa: BLE001
            pending = await self.r.xpending_range(stream, group, min=msg_id, max=msg_id, count=1)
            deliveries = pending[0]["times_delivered"] if pending else 1
            log.warning("event handler failed", stream=stream, deliveries=deliveries, error=str(e))
            if deliveries >= self.max_deliveries:
                await self.r.xadd(f"{stream}:dlq", {"data": raw, "error": str(e)[:500]})
                await self.r.xack(stream, group, msg_id)


class PubSubBus(EventBus):  # pragma: no cover - requires GCP
    def __init__(self, project: str) -> None:
        from google.cloud import pubsub_v1
        self.project = project
        self.publisher = pubsub_v1.PublisherClient(
            publisher_options=pubsub_v1.types.PublisherOptions(enable_message_ordering=False))

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        path = self.publisher.topic_path(self.project, topic.replace(".", "-"))
        future = self.publisher.publish(path, json.dumps(event, default=str).encode())
        await asyncio.wrap_future(future)  # type: ignore[arg-type]


def build_event_bus(settings, redis) -> EventBus:
    if settings.event_bus_backend == "pubsub" and settings.gcp_project:
        return PubSubBus(settings.gcp_project)
    if redis is not None:
        return RedisStreamsBus(redis)
    return InMemoryBus()
