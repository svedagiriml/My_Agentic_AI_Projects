"""Guardrail policy engine.

INPUT  (customer -> model):  prompt-injection check (block) -> PII redaction -> length cap
OUTPUT (model -> customer):  prompt-leak check (canary / system-prompt fingerprints, block)
                             -> PII redaction (the model must never echo identifiers)
"""
from __future__ import annotations

from common.metrics import GUARDRAIL_EVENTS
from common.schemas import GuardrailRequest, GuardrailResponse
from common.security import PROMPT_CANARY, SYSTEM_PROMPT_FINGERPRINTS
from services.guardrail.injection import InjectionDetector
from services.guardrail.pii import PIIRedactor

INJECTION_REFUSAL = ("I can only help with questions about our loan products. "
                     "Please rephrase your question without instructions to change how I work.")
LEAK_REFUSAL = "Sorry, I can't share that. Is there anything about our loans I can help with?"


class GuardrailEngine:
    def __init__(self, redactor: PIIRedactor, detector: InjectionDetector) -> None:
        self.redactor = redactor
        self.detector = detector

    def check(self, req: GuardrailRequest) -> GuardrailResponse:
        if req.direction == "input":
            injected, score, reasons = self.detector.is_injection(req.text)
            if injected:
                GUARDRAIL_EVENTS.labels("input", "blocked_injection").inc()
                redacted, ents = self.redactor.redact(req.text)
                return GuardrailResponse(sanitized_text=redacted, pii_found=ents, injection_detected=True,
                                         injection_score=score, injection_reasons=reasons, blocked=True,
                                         block_reason=INJECTION_REFUSAL)
            redacted, ents = self.redactor.redact(req.text)
            if ents:
                GUARDRAIL_EVENTS.labels("input", "pii_redacted").inc(len(ents))
            return GuardrailResponse(sanitized_text=redacted, pii_found=ents, injection_score=score,
                                     injection_reasons=reasons)

        # output direction
        low = req.text.lower()
        if PROMPT_CANARY.lower() in low or any(f in low for f in SYSTEM_PROMPT_FINGERPRINTS):
            GUARDRAIL_EVENTS.labels("output", "blocked_prompt_leak").inc()
            return GuardrailResponse(sanitized_text=LEAK_REFUSAL, blocked=True, block_reason="system prompt leak")
        redacted, ents = self.redactor.redact(req.text)
        if ents:
            GUARDRAIL_EVENTS.labels("output", "pii_redacted").inc(len(ents))
        return GuardrailResponse(sanitized_text=redacted, pii_found=ents)
