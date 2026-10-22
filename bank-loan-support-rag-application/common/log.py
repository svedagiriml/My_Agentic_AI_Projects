"""Structured JSON logging with trace-id propagation.

Cloud Run ships stdout to Cloud Logging; JSON lines with `severity` and
`logging.googleapis.com/trace` fields are parsed natively, so logs from all
microservices for one request can be correlated by trace id.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime

trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "severity": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        tid = trace_id_var.get()
        if tid:
            payload["trace_id"] = tid
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(service: str, level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, message: str, level: int = logging.INFO, **fields: object) -> None:
    logger.log(level, message, extra={"extra_fields": fields})
