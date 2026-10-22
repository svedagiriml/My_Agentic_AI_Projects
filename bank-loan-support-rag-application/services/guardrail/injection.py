"""Prompt-injection / jailbreak detection.

Layer 1 – normalisation: NFKC, strip zero-width & bidi control chars, collapse
          whitespace, undo common leetspeak, so "1gn0re prev1ous" and
          "i​gnore" are caught by the same rules.
Layer 2 – weighted heuristics: instruction-override phrases, role-play/jailbreak
          personas, system-prompt exfiltration, fake role/delimiter tokens
          (`<|im_start|>`, `### system`, `</context>`), encoded payloads, and
          requests to act on other customers' data.
          score = 1 - Π(1 - w_i)  (noisy-OR over the rules that fired)
Layer 3 – optional ML classifier (USE_INJECTION_CLASSIFIER=true):
          `protectai/deberta-v3-base-prompt-injection-v2` (Apache-2.0, free),
          combined as max(heuristic, model).
Defence in depth: even if something slips through, the generator prompt treats
user text and retrieved passages as data, and a canary token in the system
prompt lets the output guardrail detect a prompt leak.
"""
from __future__ import annotations

import base64
import re
import unicodedata
from dataclasses import dataclass

ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    weight: float


def _r(p: str) -> re.Pattern[str]:
    return re.compile(p, re.I | re.S)


RULES: list[Rule] = [
    Rule("override_instructions", _r(r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}"
                                     r"\b(previous|prior|above|earlier|all|your|the|system)\b.{0,30}"
                                     r"\b(instructions?|rules?|prompts?|guidelines?|directions?|context)\b"), 0.9),
    Rule("new_instructions", _r(r"\b(new|updated|real|actual)\s+(instructions?|rules?|system prompt)\b"), 0.6),
    Rule("prompt_exfiltration", _r(r"\b(reveal|show|print|repeat|output|display|leak|tell me)\b.{0,40}"
                                   r"\b(system|hidden|initial|original|developer)\s*(prompt|instructions?|message|rules)\b"), 0.85),
    Rule("verbatim_above", _r(r"\b(repeat|print|output)\b.{0,20}\b(everything|all text|words)\b.{0,20}\babove\b"), 0.7),
    Rule("persona_jailbreak", _r(r"\b(you are now|from now on you|pretend (to be|you are)|act as|roleplay as|"
                                 r"simulate)\b.{0,60}\b(dan|unfiltered|unrestricted|jailbroken|evil|no (rules|limits|"
                                 r"restrictions)|developer mode|god mode)\b"), 0.9),
    Rule("jailbreak_keywords", _r(r"\b(jailbreak|dan mode|developer mode|do anything now|uncensored mode)\b"), 0.7),
    Rule("role_tokens", _r(r"(<\|?(im_start|im_end|system|endoftext)\|?>|\[/?inst\]|^\s*#{2,}\s*(system|assistant)\b|"
                           r"</?(system|context|instructions?)>|\bBEGIN (SYSTEM|PROMPT)\b)"), 0.75),
    Rule("fake_authority", _r(r"\b(i am|this is)\s+(the|an?)?\s*(admin|administrator|developer|bank manager|"
                              r"anthropic|openai|system)\b.{0,40}\b(authori[sz]e|override|allow|permission)"), 0.6),
    Rule("other_customer_data", _r(r"\b(list|show|give|dump|export)\b.{0,30}\b(all|other|every)\b.{0,20}"
                                   r"\b(customers?|users?|accounts?|borrowers?)\b"), 0.7),
    Rule("tool_or_code_exec", _r(r"\b(execute|run|eval)\b.{0,20}\b(code|command|shell|sql|script)\b|"
                                 r"\b(drop\s+table|union\s+select|rm\s+-rf)\b"), 0.6),
    Rule("policy_evasion", _r(r"\b(without|no)\s+(any\s+)?(restrictions?|filters?|guardrails?|safety|censorship)\b"), 0.4),
]


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = ZERO_WIDTH.sub("", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _decoded_payloads(text: str) -> list[str]:
    out = []
    for blob in re.findall(r"[A-Za-z0-9+/]{24,}={0,2}", text):
        try:
            dec = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False).decode("utf-8")
            if dec.isprintable() and len(dec) > 10:
                out.append(dec)
        except Exception:  # noqa: BLE001
            continue
    return out


class InjectionDetector:
    def __init__(self, use_classifier: bool = False, model: str = "", threshold: float = 0.85,
                 heuristic_threshold: float = 0.7) -> None:
        self.threshold = threshold                    # for the ML classifier
        self.heuristic_threshold = heuristic_threshold
        self._clf = None
        if use_classifier:
            from transformers import pipeline

            self._clf = pipeline("text-classification", model=model, truncation=True, max_length=512)

    def score(self, text: str) -> tuple[float, list[str]]:
        norm = normalize(text)
        variants = [norm, norm.translate(LEET)] + [normalize(d) for d in _decoded_payloads(norm)]
        fired: dict[str, float] = {}
        for v in variants:
            for rule in RULES:
                if rule.pattern.search(v):
                    fired[rule.name] = rule.weight
        if len(variants) > 2 and fired:
            fired["encoded_payload"] = 0.5
        if ZERO_WIDTH.search(text):
            fired["hidden_characters"] = 0.3
        p_clean = 1.0
        for w in fired.values():
            p_clean *= 1 - w
        heuristic = 1 - p_clean
        model_score = 0.0
        if self._clf is not None:
            res = self._clf(norm[:2000])[0]
            model_score = res["score"] if res["label"].upper() in ("INJECTION", "LABEL_1") else 1 - res["score"]
            if model_score >= self.threshold:
                fired["ml_classifier"] = round(model_score, 3)
        return round(max(heuristic, model_score), 4), sorted(fired)

    def is_injection(self, text: str) -> tuple[bool, float, list[str]]:
        s, reasons = self.score(text)
        return s >= self.heuristic_threshold or ("ml_classifier" in reasons), s, reasons
