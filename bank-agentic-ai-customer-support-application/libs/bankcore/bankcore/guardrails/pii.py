"""PII detection & redaction.

Fast deterministic regex + checksum layer (always on) with an optional Microsoft Presidio
NER layer for names/addresses when the ``ml`` extra is installed. Covers US + Indian
identifiers since the bank operates in both regulatory contexts (GLBA/CCPA, RBI/DPDP Act).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


def luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


@dataclass
class RedactionResult:
    text: str
    entities: dict[str, int] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.entities)


_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), "[SSN REDACTED]"),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b"), "[AADHAAR REDACTED]"),
    ("PAN_IN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN REDACTED]"),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL REDACTED]"),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[IBAN REDACTED]"),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\w)"), "[PHONE REDACTED]"),
    ("CVV", re.compile(r"(?i)\b(cvv|cvc|security code)\s*[:#]?\s*\d{3,4}\b"), r"\1 [REDACTED]"),
    ("PIN", re.compile(r"(?i)\b(pin)\s*(?:is|:)?\s*\d{4,6}\b"), r"\1 [REDACTED]"),
    ("PASSWORD", re.compile(r"(?i)\b(password|passcode|passwd|otp)\s*(?:is|:|=)\s*\S+"), r"\1 [REDACTED]"),
]
_ACCOUNT_NUMBER = re.compile(r"\b\d{9,17}\b")
_PUBLIC_CONTEXT = re.compile(r"(?i)(routing|aba|swift|transit|bic)\b")
# Published identifiers that are not personal data (our own ABA routing number, etc.)
PUBLIC_NUMBERS = {"071000990"}


def _mask_cards(text: str, counts: Counter) -> str:
    def repl(m: re.Match[str]) -> str:
        raw = m.group(0)
        if luhn_valid(raw):
            counts["CARD_NUMBER"] += 1
            return f"[CARD ****{re.sub(r'[^0-9]', '', raw)[-4:]}]"
        return raw
    return _CARD.sub(repl, text)


def redact_text(text: str, *, use_ner: bool = False) -> RedactionResult:
    if not text:
        return RedactionResult(text=text or "")
    counts: Counter = Counter()
    out = _mask_cards(text, counts)
    for label, pattern, replacement in _PATTERNS:
        out, n = pattern.subn(replacement, out)
        if n:
            counts[label] += n
    def _acct(m: re.Match[str]) -> str:
        window = out[max(0, m.start() - 60):m.start()]
        if m.group(0) in PUBLIC_NUMBERS or _PUBLIC_CONTEXT.search(window):
            return m.group(0)                     # routing numbers are public, not PII
        counts["ACCOUNT_NUMBER"] += 1
        return "[ACCOUNT NUMBER REDACTED]"
    out = _ACCOUNT_NUMBER.sub(_acct, out)
    if use_ner:
        out = _presidio_pass(out, counts)
    return RedactionResult(text=out, entities=dict(counts))


_presidio = None


def _presidio_pass(text: str, counts: Counter) -> str:  # pragma: no cover - optional dependency
    global _presidio
    try:
        if _presidio is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            _presidio = (AnalyzerEngine(), AnonymizerEngine())
        analyzer, anonymizer = _presidio
        results = analyzer.analyze(text=text, language="en", entities=["PERSON", "LOCATION", "US_DRIVER_LICENSE",
                                                                         "US_PASSPORT"])
        for r in results:
            counts[r.entity_type] += 1
        return anonymizer.anonymize(text=text, analyzer_results=results).text
    except Exception:  # noqa: BLE001 — optional layer must never break the request path
        return text


def contains_pii(text: str) -> bool:
    return redact_text(text).found
