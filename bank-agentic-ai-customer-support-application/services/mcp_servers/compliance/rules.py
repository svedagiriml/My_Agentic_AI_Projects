"""Deterministic regulatory rule engine for draft responses.

Rules are code-reviewed and versioned like any other control (model risk management /
SR 11-7). An LLM never decides compliance on its own; it can only *add* findings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bankcore.guardrails.pii import redact_text


@dataclass(frozen=True)
class Rule:
    rule: str
    regulation: str
    severity: str
    pattern: re.Pattern[str]


RULES = [
    Rule("no_guaranteed_returns", "UDAAP / FINRA 2210 / SEBI advertising code", "high",
         re.compile(r"\b(guaranteed|risk[- ]free|can'?t lose|no risk)\b.{0,40}\b(return|profit|gain|growth|income)", re.I)),
    Rule("no_guaranteed_credit", "UDAAP / Fair Lending", "high",
         re.compile(r"\b(guaranteed|definitely|100%)\b.{0,25}\b(approv\w+|qualif\w+)", re.I)),
    Rule("no_protected_class_in_lending", "ECOA / Reg B / Fair Housing Act", "high",
         re.compile(r"\b(race|religion|national origin|marital status|pregnan\w+|ethnicity|caste|age)\b.{0,60}"
                    r"\b(loan|credit|mortgage|rate|approv\w+|eligib\w+)", re.I)),
    Rule("no_dispute_outcome_promise", "Reg E / UDAAP", "medium",
         re.compile(r"\b(will|definitely|guarantee)\b.{0,20}\b(get|receive|refund)\b.{0,20}\b(money|refund|back)\b", re.I)),
    Rule("no_personalised_investment_advice", "Reg BI / Investment Advisers Act", "medium",
         re.compile(r"\b(you should|i recommend|best (fund|investment) for you|put all your)\b.{0,40}"
                    r"\b(invest|buy|fund|stock|portfolio)", re.I)),
    Rule("no_credentials_request", "GLBA Safeguards / FFIEC authentication guidance", "high",
         re.compile(r"\b(send|share|provide|tell|give)\b.{0,25}\b(your )?(password|pin|otp|one[- ]time code|cvv)\b", re.I)),
]

DISCLOSURE_TRIGGERS = [
    (re.compile(r"\b(fund|mutual fund|ira|portfolio|stocks?|bonds?|invest\w*)\b", re.I),
     "Investment products are not FDIC insured, are not bank guaranteed and may lose value."),
    (re.compile(r"\b(apy|cd rate|certificate of deposit|savings rate)\b", re.I),
     "APY is accurate as of the date shown and may change; penalties may apply for early CD withdrawal."),
    (re.compile(r"\b(apr|mortgage|auto loan|personal loan|heloc|interest rate)\b", re.I),
     "Rates are indicative, subject to credit approval and may change. This is not a commitment to lend."),
    (re.compile(r"\b(exchange rate|fx rate|wire)\b", re.I),
     "Exchange rates are indicative; the final rate is set at execution."),
]


def audit(text: str) -> dict:
    findings = []
    for r in RULES:
        m = r.pattern.search(text)
        if m:
            findings.append({"rule": r.rule, "regulation": r.regulation, "severity": r.severity,
                             "excerpt": m.group(0)[:200]})
    pii = redact_text(text)
    for entity in ("SSN", "CARD_NUMBER", "AADHAAR", "PAN_IN", "PASSWORD", "CVV", "PIN", "ACCOUNT_NUMBER"):
        if entity in pii.entities:
            findings.append({"rule": f"no_raw_{entity.lower()}", "regulation": "GLBA / PCI-DSS / GDPR / DPDP",
                             "severity": "high", "excerpt": f"{pii.entities[entity]} occurrence(s)"})
    disclosures = [d for p, d in DISCLOSURE_TRIGGERS if p.search(text)]
    compliant = not any(f["severity"] == "high" for f in findings)
    return {"compliant": compliant, "findings": findings, "required_disclosures": disclosures}
