"""PII detection & redaction tuned for Indian banking text.

Detectors (regex + checksum/context validation to keep false positives low, so
"Rs.30 lakh", "0.35%" or "240 months" are never redacted):

| Entity        | Rule                                                                |
|---------------|---------------------------------------------------------------------|
| EMAIL         | RFC-ish address                                                     |
| CARD          | 13-19 digits (spaces/dashes ok) **and Luhn checksum valid**         |
| AADHAAR       | 12 digits, first 2-9, optional spaces **and Verhoeff checksum**     |
| PAN           | AAAAA9999A with valid 4th-char holder type (P,C,H,F,A,T,B,L,J,G)    |
| IFSC          | AAAA0XXXXXX                                                         |
| UPI_ID        | handle@psp (no dot in domain, e.g. name@oksbi)                      |
| PHONE         | Indian mobile: optional +91/0, starts 6-9, 10 digits                |
| BANK_ACCOUNT  | 9-18 digits within 25 chars after "account/a/c/acct no"             |
| PASSPORT      | A1234567 near the word passport                                     |
| IP_ADDRESS    | IPv4                                                                |
| PERSON/LOC    | optional Microsoft Presidio (spaCy NER) when USE_PRESIDIO=true      |

Every distinct value gets a stable placeholder within a request (`<PAN_1>`), so
the LLM can still reason ("the PAN you shared") without seeing the value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from common.schemas import PIIEntity

# --------------------------------------------------------------------------- checksums
_VERHOEFF_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
               [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
               [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
               [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VERHOEFF_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
               [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
               [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_valid(num: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def luhn_valid(num: str) -> bool:
    total, alt = 0, False
    for ch in reversed(num):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


# --------------------------------------------------------------------------- detectors
@dataclass(frozen=True)
class Detector:
    entity: str
    pattern: re.Pattern[str]
    priority: int
    group: int = 0
    validate: object = None   # callable(str) -> bool


DETECTORS: list[Detector] = [
    Detector("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), 100),
    Detector("CARD", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), 95,
             validate=lambda s: 13 <= len(_digits(s)) <= 19 and luhn_valid(_digits(s))),
    Detector("AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"), 90,
             validate=lambda s: len(_digits(s)) == 12 and verhoeff_valid(_digits(s))),
    Detector("PAN", re.compile(r"\b[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]\b", re.I), 85),
    Detector("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), 80),
    Detector("UPI_ID", re.compile(r"\b[\w.-]{2,}@[a-z]{2,}\b(?!\.\w)", re.I), 75),
    Detector("BANK_ACCOUNT", re.compile(
        r"(?:account|a/c|acct|ac)\.?\s*(?:no\.?|number|#)?\s*(?:is|:|-)?\s*(\d[\d -]{7,20}\d)", re.I), 70, group=1,
        validate=lambda s: 9 <= len(_digits(s)) <= 18),
    Detector("PHONE", re.compile(r"(?<![\d+])(?:\+?91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)"), 65),
    Detector("PASSPORT", re.compile(r"passport\s*(?:no\.?|number)?\s*(?:is|:)?\s*([A-Z][1-9]\d{5}[1-9])\b", re.I),
             60, group=1),
    Detector("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), 55),
]


@dataclass
class _Span:
    start: int
    end: int
    entity: str
    priority: int
    value: str


class PIIRedactor:
    def __init__(self, use_presidio: bool = False) -> None:
        self._presidio = None
        if use_presidio:
            from presidio_analyzer import AnalyzerEngine

            self._presidio = AnalyzerEngine()

    def _find(self, text: str) -> list[_Span]:
        spans: list[_Span] = []
        for d in DETECTORS:
            for m in d.pattern.finditer(text):
                val = m.group(d.group)
                if d.validate is not None and not d.validate(val):  # type: ignore[operator]
                    continue
                spans.append(_Span(m.start(d.group), m.end(d.group), d.entity, d.priority, val))
        if self._presidio is not None:
            for r in self._presidio.analyze(text=text, language="en",
                                            entities=["PERSON", "LOCATION", "DATE_TIME", "NRP"]):
                if r.score >= 0.6 and r.entity_type != "DATE_TIME":
                    spans.append(_Span(r.start, r.end, r.entity_type, 40, text[r.start:r.end]))
        # resolve overlaps: higher priority wins, then longer span
        spans.sort(key=lambda s: (-s.priority, -(s.end - s.start)))
        chosen: list[_Span] = []
        for s in spans:
            if all(s.end <= c.start or s.start >= c.end for c in chosen):
                chosen.append(s)
        return sorted(chosen, key=lambda s: s.start)

    def redact(self, text: str) -> tuple[str, list[PIIEntity]]:
        spans = self._find(text)
        if not spans:
            return text, []
        counters: dict[str, int] = {}
        placeholders: dict[tuple[str, str], str] = {}
        out, entities, cursor, shift = [], [], 0, 0
        for s in spans:
            key = (s.entity, re.sub(r"\W", "", s.value).lower())
            if key not in placeholders:
                counters[s.entity] = counters.get(s.entity, 0) + 1
                placeholders[key] = f"<{s.entity}_{counters[s.entity]}>"
            ph = placeholders[key]
            out.append(text[cursor:s.start])
            out.append(ph)
            new_start = s.start + shift
            entities.append(PIIEntity(entity_type=s.entity, start=new_start, end=new_start + len(ph), placeholder=ph))
            shift += len(ph) - (s.end - s.start)
            cursor = s.end
        out.append(text[cursor:])
        return "".join(out), entities
