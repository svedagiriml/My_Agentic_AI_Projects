"""Prompt-injection detection for direct (user) and indirect (tool / RAG / memory) content.

Layered defence:
1. Normalisation — NFKC, strip zero-width/bidi control chars, collapse whitespace
   (defeats unicode smuggling and spacing obfuscation).
2. Weighted heuristic signatures (instruction override, role hijack, prompt exfiltration,
   tool-call spoofing, encoded payloads, financial coercion).
3. Optional free ML classifier (``protectai/deberta-v3-base-prompt-injection-v2``) when the
   ``ml`` extra is installed and ``PI_CLASSIFIER_ENABLED=1``.
4. Spotlighting — untrusted text is always wrapped in delimiters by the ContextBuilder.
"""
from __future__ import annotations

import base64
import os
import re
import unicodedata
from dataclasses import dataclass, field

_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")

# (name, weight, pattern)
_SIGNATURES: list[tuple[str, float, re.Pattern[str]]] = [
    ("override_instructions", 0.9, re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|your|system)\b"
        r".{0,30}\b(instructions?|rules?|prompts?|guidelines|polic(y|ies)|directives?)", re.I | re.S)),
    ("new_instructions", 0.6, re.compile(r"\b(new|updated|real|actual)\s+(instructions?|system prompt|rules)\b", re.I)),
    ("role_hijack", 0.7, re.compile(
        r"\b(you are now|act as|pretend (to be|you are)|roleplay as|from now on you|switch to)\b.{0,60}"
        r"\b(admin|developer|dan|unrestricted|jailbroken|root|god mode|no (rules|limits|restrictions))", re.I | re.S)),
    ("jailbreak_terms", 0.6, re.compile(r"\b(jailbreak|DAN mode|developer mode|do anything now)\b", re.I)),
    ("prompt_exfiltration", 0.8, re.compile(
        r"\b(reveal|show|print|repeat|output|tell me|what (is|are))\b.{0,30}\b(system prompt|your instructions|"
        r"hidden (prompt|rules)|initial prompt|the text above|canary)", re.I | re.S)),
    ("role_markers", 0.6, re.compile(r"(^|\n)\s*(system|assistant|developer)\s*:|<\|?(im_start|system|endoftext)\|?>|"
                                      r"\[/?INST\]|<</?SYS>>", re.I)),
    ("tool_spoofing", 0.7, re.compile(r"<\s*/?\s*(tool_call|function_call|tool_result)|\"tool_calls\"\s*:", re.I)),
    ("privilege_claim", 0.5, re.compile(
        r"\b(i am|i'm|this is)\s+(an?|the)\s+(admin|administrator|bank (manager|officer|employee)|developer|"
        r"anthropic|openai|security team)\b", re.I)),
    ("approval_bypass", 0.8, re.compile(
        r"\b(skip|bypass|without|no need for|don't need)\b.{0,25}\b(approval|verification|confirmation|mfa|"
        r"authentication|human review)\b", re.I | re.S)),
    ("mass_exfil", 0.7, re.compile(r"\b(all|every|other)\s+(customers?|users?|accounts?)('s)?\b.{0,40}"
                                   r"\b(balance|data|details|numbers|records|transactions)", re.I | re.S)),
    ("coercive_transfer", 0.5, re.compile(r"\b(transfer|send|wire)\b.{0,30}\b(all|entire|everything|maximum)\b.{0,30}"
                                          r"\b(to|into)\b.{0,20}\b(account|wallet|crypto)", re.I | re.S)),
    ("memory_poison", 0.6, re.compile(r"\b(remember|memorize|store|save)\b.{0,30}\b(that|this)\b.{0,40}\b(always|never|"
                                      r"from now on|in future|every time)\b", re.I | re.S)),
]

_B64 = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _ZERO_WIDTH.sub("", text)
    return re.sub(r"[ \t]+", " ", text)


@dataclass
class InjectionVerdict:
    score: float
    is_injection: bool
    signals: list[str] = field(default_factory=list)
    normalized_text: str = ""


class PromptInjectionDetector:
    def __init__(self, threshold: float = 0.7, use_classifier: bool | None = None) -> None:
        self.threshold = threshold
        self.use_classifier = (os.getenv("PI_CLASSIFIER_ENABLED", "0") == "1") if use_classifier is None \
            else use_classifier
        self._clf = None

    def _classifier_score(self, text: str) -> float:  # pragma: no cover - optional model
        try:
            if self._clf is None:
                from transformers import pipeline
                self._clf = pipeline("text-classification",
                                     model="protectai/deberta-v3-base-prompt-injection-v2", truncation=True)
            res = self._clf(text[:2000])[0]
            return float(res["score"]) if res["label"].upper() == "INJECTION" else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def scan(self, text: str, *, untrusted_source: bool = False) -> InjectionVerdict:
        norm = normalize(text)
        signals: list[str] = []
        score = 0.0
        for name, weight, pattern in _SIGNATURES:
            if pattern.search(norm):
                signals.append(name)
                score = 1 - (1 - score) * (1 - weight)      # noisy-OR combination
        if _ZERO_WIDTH.search(text or ""):
            signals.append("hidden_unicode")
            score = 1 - (1 - score) * 0.6
        for blob in _B64.findall(norm)[:3]:
            try:
                decoded = base64.b64decode(blob + "==", validate=False).decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                continue
            if decoded and any(p.search(decoded) for _, w, p in _SIGNATURES if w >= 0.6):
                signals.append("encoded_payload")
                score = 1 - (1 - score) * 0.2
        if self.use_classifier:
            cs = self._classifier_score(norm)
            if cs > 0.5:
                signals.append("ml_classifier")
            score = max(score, cs)
        threshold = self.threshold - 0.15 if untrusted_source else self.threshold
        return InjectionVerdict(score=round(score, 3), is_injection=score >= threshold, signals=signals,
                                normalized_text=norm)


def neutralize_untrusted(text: str) -> str:
    """Defang instruction-like markup inside untrusted data before it enters a prompt."""
    text = normalize(text)
    text = re.sub(r"<\s*/?\s*(system|assistant|tool_call|function_call|untrusted_data)[^>]*>", "[removed-tag]", text,
                  flags=re.I)
    text = re.sub(r"(^|\n)\s*(system|assistant|developer)\s*:", r"\1[role-marker-removed]:", text, flags=re.I)
    return text
