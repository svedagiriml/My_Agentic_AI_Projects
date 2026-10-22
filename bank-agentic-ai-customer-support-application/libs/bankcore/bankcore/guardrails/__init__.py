"""Guardrail pipeline facade used by the supervisor's input/output nodes."""
from __future__ import annotations

from dataclasses import dataclass, field

from bankcore.guardrails.injection import PromptInjectionDetector, neutralize_untrusted  # noqa: F401
from bankcore.guardrails.memory_guard import MemoryGuard  # noqa: F401
from bankcore.guardrails.output_guard import CANARY, guard_output, grounding_check  # noqa: F401
from bankcore.guardrails.pii import redact_text


@dataclass
class InputGuardResult:
    allowed: bool
    sanitized_text: str
    pii_entities: dict[str, int] = field(default_factory=dict)
    injection_score: float = 0.0
    signals: list[str] = field(default_factory=list)
    reason: str = ""


_OFF_TOPIC = ("write me a poem", "write code", "homework", "recipe for")


class InputGuard:
    """Direct prompt-injection + PII redaction + topical scope for customer input."""

    def __init__(self, detector: PromptInjectionDetector | None = None) -> None:
        self.detector = detector or PromptInjectionDetector()

    def check(self, text: str) -> InputGuardResult:
        verdict = self.detector.scan(text)
        # PII is redacted BEFORE the text reaches any LLM, cache, log or memory store.
        red = redact_text(verdict.normalized_text)
        if verdict.is_injection:
            return InputGuardResult(False, red.text, red.entities, verdict.score, verdict.signals,
                                    "prompt_injection")
        lowered = red.text.lower()
        if any(p in lowered for p in _OFF_TOPIC):
            return InputGuardResult(False, red.text, red.entities, verdict.score, ["off_topic"], "off_topic")
        return InputGuardResult(True, red.text, red.entities, verdict.score, verdict.signals)
