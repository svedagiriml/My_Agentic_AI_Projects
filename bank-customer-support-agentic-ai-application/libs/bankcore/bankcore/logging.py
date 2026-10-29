"""Structured JSON logging with mandatory PII scrubbing.

Every log event passes through ``_redact_processor`` so card numbers, SSNs,
e-mails, phone numbers and secrets never reach Cloud Logging.
"""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

import structlog

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_SENSITIVE_KEYS = {"password", "token", "authorization", "secret", "delegation_token", "approval_token",
                   "api_key", "cvv", "pin", "ssn"}


def _redact_processor(_logger, _method, event_dict: dict) -> dict:
    from bankcore.guardrails.pii import redact_text  # local import avoids a cycle

    for key in list(event_dict.keys()):
        val = event_dict[key]
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "***"
        elif isinstance(val, str):
            event_dict[key] = redact_text(val).text
    return event_dict


def _context_processor(_logger, _method, event_dict: dict) -> dict:
    if (rid := request_id_var.get()) is not None:
        event_dict.setdefault("request_id", rid)
    if (tid := trace_id_var.get()) is not None:
        event_dict.setdefault("trace_id", tid)
    return event_dict


def configure_logging(service_name: str, level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _context_processor,
            _redact_processor,
            structlog.processors.format_exc_info,
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: str | None = None):
    return structlog.get_logger(name)
