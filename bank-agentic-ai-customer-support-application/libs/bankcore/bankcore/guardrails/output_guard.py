"""Output-side guardrails applied to the final customer-facing answer."""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

from bankcore.guardrails.pii import redact_text

# Canary token embedded in every system prompt. If it ever appears in an output, the model
# has been coerced into leaking its instructions -> block.
CANARY = f"CANARY-{secrets.token_hex(6)}"

_REASONING_LEAK = re.compile(r"(?im)^\s*(thought|reasoning|chain of thought|action input|observation)\s*:.*$")
_MONEY = re.compile(r"(?<![\w.])(?:[$₹€£]\s?)(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)(?![\d,])")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")


@dataclass
class OutputGuardResult:
    text: str
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)
    ungrounded_figures: list[str] = field(default_factory=list)


def _normalise_number(s: str) -> str:
    s = s.replace(",", "")
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return s


def extract_figures(text: str) -> set[str]:
    figs = {_normalise_number(m.group(1)) for m in _MONEY.finditer(text)}
    figs |= {_normalise_number(m.group(1)) + "%" for m in _PERCENT.finditer(text)}
    return figs


def evidence_figures(evidence: list[str]) -> set[str]:
    found: set[str] = set()
    for e in evidence:
        for m in re.finditer(r"-?\d+(?:,\d{3})*(?:\.\d+)?", e):
            n = _normalise_number(m.group(0).lstrip("-"))
            found.add(n)
            found.add(n + "%")
    return found


def grounding_check(answer: str, evidence: list[str]) -> list[str]:
    """Return monetary/percentage figures in ``answer`` that do not appear in any evidence."""
    ev = evidence_figures(evidence)
    return sorted(f for f in extract_figures(answer) if f not in ev)


def guard_output(text: str, *, evidence: list[str] | None = None) -> OutputGuardResult:
    res = OutputGuardResult(text=text)
    if CANARY in text:
        res.blocked = True
        res.reasons.append("system_prompt_leak")
        res.text = "I'm sorry, I can't share that."
        return res
    cleaned = _REASONING_LEAK.sub("", text).strip()
    if cleaned != text.strip():
        res.reasons.append("reasoning_stripped")
    red = redact_text(cleaned)
    if red.found:
        res.reasons.append(f"pii_redacted:{sorted(red.entities)}")
    res.text = red.text
    if evidence is not None:
        res.ungrounded_figures = grounding_check(res.text, evidence)
        if res.ungrounded_figures:
            res.reasons.append("ungrounded_figures")
    return res
