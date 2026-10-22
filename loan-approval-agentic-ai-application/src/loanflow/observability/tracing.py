"""LangFuse tracing (self-hostable, free) with a no-op fallback.

Trace model
-----------
  trace  = one loan run (session_id = run_id, user_id = officer)
  span   = supervisor step / subagent execution / MCP tool call
  generation = each LLM call (model, prompt, completion, token usage, cost)

All inputs/outputs are PII-redacted *before* they are sent to LangFuse.
W3C `traceparent` is propagated over HTTP between Cloud Run services so the
supervisor and subagent spans stitch into one trace.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from loanflow.common.config import get_settings
from loanflow.common.logging import get_logger
from loanflow.security.guardrails import redact_pii

log = get_logger(__name__)
_client: Any = None
_disabled = False


def _redact(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact_pii(obj).text
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def get_langfuse():
    global _client, _disabled
    if _client is not None or _disabled:
        return _client
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        _disabled = True
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host)
    except Exception as exc:  # noqa: BLE001
        log.warning("langfuse unavailable, tracing disabled", extra={"extra_fields": {"error": str(exc)}})
        _disabled = True
    return _client


class _NoopObs:
    def update(self, **_: Any) -> None: ...
    def update_trace(self, **_: Any) -> None: ...


@contextlib.contextmanager
def span(name: str, *, input: Any = None, metadata: dict | None = None) -> Iterator[Any]:
    lf = get_langfuse()
    if lf is None:
        yield _NoopObs()
        return
    try:
        cm = lf.start_as_current_span(name=name, input=_redact(input), metadata=metadata or {})
    except Exception:  # noqa: BLE001
        yield _NoopObs()
        return
    with cm as obs:
        yield obs


@contextlib.contextmanager
def generation(name: str, *, model: str, input: Any, model_parameters: dict | None = None) -> Iterator[Any]:
    lf = get_langfuse()
    if lf is None:
        yield _NoopObs()
        return
    try:
        cm = lf.start_as_current_generation(
            name=name, model=model, input=_redact(input), model_parameters=model_parameters or {}
        )
    except Exception:  # noqa: BLE001
        yield _NoopObs()
        return
    with cm as obs:
        yield obs


def tag_trace(*, run_id: str, user_id: str | None = None, tags: list[str] | None = None, **meta: Any) -> None:
    lf = get_langfuse()
    if lf is None:
        return
    with contextlib.suppress(Exception):
        lf.update_current_trace(session_id=run_id, user_id=user_id, tags=tags or [], metadata=_redact(meta))


def flush() -> None:
    lf = get_langfuse()
    if lf is not None:
        with contextlib.suppress(Exception):
            lf.flush()
