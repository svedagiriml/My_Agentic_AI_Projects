"""Message verification at handoff.

Each envelope is serialised canonically (sorted keys, no whitespace), hashed with
SHA-256 and the hash is HMAC-signed with a shared key (from Secret Manager in prod).

At every handoff the receiver:
  1. re-parses the JSON with the pydantic model          -> structure/type check
  2. recomputes the checksum and compares                 -> tamper / corruption check
  3. verifies the HMAC signature (constant-time compare)  -> authenticity check
  4. (supervisor only) validates `output` against the node's typed output schema
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from loanflow.common.config import get_settings
from loanflow.common.exceptions import MessageIntegrityError
from loanflow.common.schemas import NODE_OUTPUT_SCHEMAS, AgentResult, Envelope

E = TypeVar("E", bound=Envelope)
_SIG_FIELDS = {"checksum", "signature"}


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _body(msg: Envelope) -> str:
    return canonical_json(msg.model_dump(mode="json", exclude=_SIG_FIELDS))


def _hmac(checksum: str, key: str | None = None) -> str:
    key = key or get_settings().message_hmac_key
    return hmac.new(key.encode(), checksum.encode(), hashlib.sha256).hexdigest()


def sign_message(msg: E) -> E:
    checksum = sha256_hex(_body(msg))
    return msg.model_copy(update={"checksum": checksum, "signature": _hmac(checksum)})


def verify_message(msg: Envelope) -> None:
    if not msg.checksum or not msg.signature:
        raise MessageIntegrityError("message is unsigned", details={"message_id": msg.message_id})
    expected = sha256_hex(_body(msg))
    if not hmac.compare_digest(expected, msg.checksum):
        raise MessageIntegrityError("checksum mismatch (payload altered)", details={"message_id": msg.message_id})
    if not hmac.compare_digest(_hmac(msg.checksum), msg.signature):
        raise MessageIntegrityError("invalid signature", details={"message_id": msg.message_id})


def parse_and_verify(model: type[E], raw: str | bytes | dict) -> E:
    try:
        msg = model.model_validate_json(raw) if isinstance(raw, (str, bytes)) else model.model_validate(raw)
    except ValidationError as exc:
        raise MessageIntegrityError("schema validation failed", details={"errors": exc.errors()[:5]}) from exc
    verify_message(msg)
    return msg


def validate_node_output(node: str, result: AgentResult) -> BaseModel | None:
    """Strictly type-check a subagent's output against that node's contract."""
    if result.status != "success":
        return None
    schema = NODE_OUTPUT_SCHEMAS.get(node)
    if schema is None:
        raise MessageIntegrityError(f"no output schema registered for node '{node}'")
    if result.sender != node:
        raise MessageIntegrityError(f"sender mismatch: expected {node}, got {result.sender}")
    try:
        return schema.model_validate(result.output)
    except ValidationError as exc:
        raise MessageIntegrityError(
            f"output of {node} violates its schema", details={"errors": exc.errors()[:5]}
        ) from exc
