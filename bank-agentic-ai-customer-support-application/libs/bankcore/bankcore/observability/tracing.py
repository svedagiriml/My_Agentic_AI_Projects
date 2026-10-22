"""Langfuse tracing (self-hosted, open source) — one trace spans all microservices.

The supervisor creates the trace id; it travels inside every HandoffEnvelope so sub-agent
spans (LLM generations, MCP tool calls) attach to the same trace. If Langfuse is not
configured or unreachable, everything degrades to no-ops — observability must never take
the request path down.
"""
from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from typing import Any

from bankcore.config import get_settings
from bankcore.logging import get_logger

log = get_logger(__name__)
_client = None
_disabled = False


def new_trace_id() -> str:
    # Langfuse (OTel) trace ids are 32 lowercase hex chars
    return uuid.uuid4().hex


def get_langfuse():
    global _client, _disabled
    if _disabled:
        return None
    if _client is not None:
        return _client
    s = get_settings()
    if not (s.langfuse_enabled and s.langfuse_public_key and s.langfuse_secret_key):
        _disabled = True
        return None
    try:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", s.langfuse_public_key.get_secret_value())
        os.environ.setdefault("LANGFUSE_SECRET_KEY", s.langfuse_secret_key.get_secret_value())
        os.environ.setdefault("LANGFUSE_HOST", s.langfuse_host)
        from langfuse import Langfuse

        _client = Langfuse(public_key=s.langfuse_public_key.get_secret_value(),
                           secret_key=s.langfuse_secret_key.get_secret_value(), host=s.langfuse_host)
        return _client
    except Exception as e:  # noqa: BLE001
        log.warning("langfuse disabled", error=str(e))
        _disabled = True
        return None


def langfuse_callbacks(trace_id: str, **_: Any) -> list:
    """LangChain callback handlers bound to ``trace_id`` (empty list when disabled)."""
    if get_langfuse() is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler(trace_context={"trace_id": trace_id})]
    except Exception:  # noqa: BLE001
        try:  # Langfuse SDK v2 fallback
            from langfuse.callback import CallbackHandler as V2Handler  # type: ignore
            return [V2Handler(trace_id=trace_id)]
        except Exception:  # noqa: BLE001
            return []


@contextmanager
def span(name: str, trace_id: str, **attrs: Any):
    """Manual span for non-LLM work (MCP tool call, DB, cache). No-op when disabled."""
    lf = get_langfuse()
    if lf is None:
        yield None
        return
    try:
        cm = lf.start_as_current_observation(name=name, as_type="span",
                                             trace_context={"trace_id": trace_id}, input=attrs or None)
    except Exception:  # noqa: BLE001
        yield None
        return
    with cm as obs:
        yield obs


def score(trace_id: str, name: str, value: float, comment: str | None = None) -> None:
    """Attach an evaluation score (guardrail hits, eval metrics, HITL verdicts) to a trace."""
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.create_score(trace_id=trace_id, name=name, value=value, comment=comment)
    except Exception as e:  # noqa: BLE001
        log.debug("langfuse score failed", error=str(e))


def flush() -> None:
    lf = get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:  # noqa: BLE001
            pass
