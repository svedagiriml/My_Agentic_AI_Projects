"""Deterministic, rule-based stand-ins for the LLMs (no network/GPU).

Used by integration tests and by ``eval/run_eval.py --offline`` so the whole multi-agent
pipeline (planning, ReAct tool calls, HITL, compliance, memory) can be exercised in CI.
They mimic the *protocol* of the real models (JSON plans, tool calls, Thought: lines) — not
their intelligence. Real-model evaluation uses ``--online`` with Ollama/Groq.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import date, timedelta

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from bankcore.llm.fake import ScriptedChatModel

_USER = re.compile(r'## 7\. USER INPUT\n<untrusted_data source="customer">\n(.*?)\n</untrusted_data>', re.S)
_FACTS = re.compile(r'## 5\. RETRIEVED FACTS.*?<untrusted_data source="retrieval">\n(.*?)\n</untrusted_data>', re.S)


def _user_text(messages: list[BaseMessage]) -> str:
    for m in messages:
        if isinstance(m, HumanMessage):
            hit = _USER.search(str(m.content))
            if hit:
                return hit.group(1)
    return str(messages[-1].content)


def _system(messages: list[BaseMessage]) -> str:
    return next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")


def _first(pattern: str, text: str, default: str | None = None) -> str | None:
    m = re.search(pattern, text)
    return m.group(0) if m else default


def _amount(text: str) -> float | None:
    m = re.search(r"[$₹]\s?([\d,]+(?:\.\d{1,2})?)|\b([\d,]+(?:\.\d{1,2})?)\s?(?:dollars|usd)\b", text, re.I)
    if not m:
        return None
    return float((m.group(1) or m.group(2)).replace(",", ""))


def _call(name: str, args: dict, thought: str) -> AIMessage:
    return AIMessage(content=f"Thought: {thought}", tool_calls=[{"name": name, "args": args,
                                                                 "id": f"call_{uuid.uuid4().hex[:8]}",
                                                                 "type": "tool_call"}])


# ------------------------------------------------------------------ supervisor --
def supervisor_responder(messages: list[BaseMessage]) -> AIMessage:
    from supervisor.app.fallback_router import keyword_plan  # local import (service package)

    system = _system(messages)
    if "You are the Supervisor of Horizon Trust" in system:
        text = _user_text(messages)
        plan = keyword_plan(text).model_dump(mode="json")
        plan["reasoning"] = ["Restate the request.", f"Intents found: {[t['agent'] for t in plan['tasks']]}.",
                             "Map intents to specialist agents.", "Preserve customer order; fraud first.",
                             "No clarification needed.", "Confidence high."]
        plan["confidence"] = 0.9
        return AIMessage(content=json.dumps(plan))
    if "composing the final reply" in system:
        facts = _FACTS.search(str(messages[-1].content))
        lines = [re.sub(r"^- Request \d+ \([a-z]+, status=\w+\): ", "", line)
                 for line in (facts.group(1).splitlines() if facts else [])]
        return AIMessage(content="\n\n".join(x for x in lines if x.strip()))
    if "Rewrite the draft reply" in system:
        draft = str(messages[-1].content).split("Draft:\n", 1)[-1]
        return AIMessage(content=re.sub(r"(?i)guaranteed", "potential", draft))
    return AIMessage(content="OK")


# -------------------------------------------------------------------- agents --
def _observations(messages: list[BaseMessage]) -> list[dict]:
    out = []
    for m in messages:
        if isinstance(m, ToolMessage):
            raw = re.sub(r"^<untrusted_data[^>]*>|</untrusted_data>$", "", str(m.content))
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                out.append({"ok": False, "raw": raw})
    return out


def _final_from(agent: str, obs: list[dict]) -> str:
    last = obs[-1] if obs else {}
    if not last.get("ok"):
        status = last.get("status") or (last.get("error") or {}).get("message") if isinstance(last.get("error"), dict) \
            else last.get("error")
        return f"I wasn't able to complete that: {status or 'the request could not be processed'}."
    d = last.get("data", {})
    if "available_balance" in d:
        return (f"Your {d['account_type']} account {d['account_id']} has an available balance of "
                f"${d['available_balance']} {d['currency']}.")
    if "transactions" in d:
        items = "; ".join(f"{t['description']} ${t['amount']}" for t in d["transactions"][:3])
        return f"Here are your most recent transactions: {items}."
    if "masked_pan" in d:
        return f"Card {d['masked_pan']} is now {d['status']}. {d['message']}"
    if "download_url" in d:
        return f"Your {d['statement_period']} statement is ready: {d['download_url']} (link expires in 15 minutes)."
    if "previous_limit" in d:
        return f"Your daily ATM limit on {d['card_id']} changed from ${d['previous_limit']} to ${d['new_limit']}."
    if "application_id" in d:
        return f"Application {d['application_id']} is in {d['status']}. Next step: {d['next_step']}"
    if "order_id" in d and "new_card_masked_pan" in d:
        return f"Replacement card {d['new_card_masked_pan']} will arrive by {d['estimated_delivery']}."
    if "transfer_id" in d:
        return f"Transfer {d['transfer_id']} of ${d['amount']} is {d['status']}."
    if "monthly_payment" in d:
        return f"The estimated monthly payment is ${d['monthly_payment']} with total interest ${d['total_interest']}."
    if "apr_from" in d:
        return f"APR ranges from {d['apr_from']}% to {d['apr_to']}%. {d['disclosure']}"
    if "chunks" in d:
        if not d["chunks"]:
            return "I couldn't find that in our policies."
        c = d["chunks"][0]
        return f"{c['content']} (Source: {c['title']})"
    if "results" in d:
        b = d["results"][0]
        return f"The nearest location is {b['name']} at {b['address']} ({b['hours']})."
    if "rates" in d:
        best = max(d["rates"], key=lambda r: float(r["apy"]))
        return f"Our best CD rate is {best['apy']}% APY for {best['term_months']} months."
    if "total_value" in d:
        return f"Your {d['account_type']} is valued at ${d['total_value']}."
    if "sessions" in d:
        return "Active sessions: " + "; ".join(f"{s['device']} in {s['location']}" for s in d["sessions"]) + "."
    if "rate" in d:
        return f"1 {d['base_currency']} = {d['rate']} {d['target_currency']} (indicative)."
    if "case_id" in d:
        return f"Fraud case {d['case_id']} opened for {d['transaction_id']}."
    if "valid_checksum" in d:
        return f"Routing number {d['routing_num']} is {'valid' if d['valid_checksum'] else 'invalid'}."
    return "Done."


def agent_responder(agent: str):
    def respond(messages: list[BaseMessage]) -> AIMessage:
        text = _user_text(messages)
        low = text.lower()
        obs = _observations(messages)
        if agent == "compliance":
            if not obs:
                draft = text.split("\n", 1)[-1]
                return _call("redact_pii", {"text_payload": draft}, "First redact PII from the draft.")
            if len(obs) == 1:
                red = obs[0].get("data", {}).get("redacted_text", "")
                return _call("audit_compliance_check", {"agent_response_text": red or "empty"},
                             "Now audit the redacted draft.")
            return AIMessage(content=json.dumps({"compliant": True, "violations": [], "redacted_text": "",
                                                 "required_disclosures": []}))
        if obs:
            return AIMessage(content="Thought: I have what I need.\n" + _final_from(agent, obs))
        acc = _first(r"ACC-\d{6}", text)
        accs = re.findall(r"ACC-\d{6}", text)
        card = _first(r"CRD-\d{6}", text, "CRD-100001")
        if agent == "account":
            if "transaction" in low or "spent" in low or "history" in low:
                end = date(2026, 10, 19)
                return _call("fetch_transaction_history", {"account_id": acc or "ACC-100001", "date_range": {
                    "start": str(end - timedelta(days=30)), "end": str(end)}, "limit": 5}, "Fetch recent transactions.")
            if "statement" in low:
                return _call("download_monthly_statement", {"account_id": acc or "ACC-100001",
                                                            "statement_period": "2026-09"}, "Get the statement link.")
            return _call("get_account_balance", {"account_id": acc or "ACC-100001"}, "Look up the balance.")
        if agent == "cards":
            if "lost" in low or "stolen" in low:
                return _call("report_lost_stolen_card", {"card_id": card}, "Block the lost card immediately.")
            if "limit" in low:
                return _call("update_atm_withdrawal_limit", {"card_id": card, "new_limit": _amount(text) or 800},
                             "Change the ATM limit.")
            return _call("freeze_or_unfreeze_card", {"card_id": card,
                                                     "status": "unfreeze" if "unfreeze" in low else "freeze"},
                         "Freeze the card.")
        if agent == "payments":
            if "rate" in low or "exchange" in low:
                cur = re.findall(r"\b[A-Z]{3}\b", text)
                cur = [c for c in cur if c not in ("ACC", "CRD")]
                return _call("get_exchange_rates", {"base_currency": cur[0] if cur else "USD",
                                                    "target_currency": cur[1] if len(cur) > 1 else "INR"},
                             "Fetch the FX rate.")
            if "routing" in low:
                return _call("verify_routing_number", {"routing_num": _first(r"\b\d{9}\b", text, "000000000")},
                             "Validate the routing number.")
            if len(accs) < 2 or _amount(text) is None:
                return AIMessage(content="Which account should I move the money from and to, and how much?")
            return _call("initiate_internal_transfer", {"source_acc": accs[0], "dest_acc": accs[1],
                                                        "amount": _amount(text)}, "Initiate the transfer.")
        if agent == "fraud":
            txn = _first(r"TXN-[A-Z0-9]+", text)
            if txn:
                return _call("flag_fraudulent_transaction", {"transaction_id": txn,
                                                             "reason": "Customer does not recognise this charge"},
                             "Flag the disputed transaction.")
            if "lock" in low:
                uid = _first(r"USR-\d{4}", str(messages[1].content), "USR-1001")
                return _call("lock_user_profile", {"user_id": uid}, "Lock the profile to contain takeover.")
            uid = _first(r"USR-\d{4}", str(messages[1].content), "USR-1001")
            return _call("fetch_active_sessions", {"user_id": uid}, "Review active sessions.")
        if agent == "loans":
            if "app-" in low:
                return _call("fetch_loan_application_status", {"application_id": _first(r"APP-\d{6}", text)},
                             "Check application status.")
            if "payment" in low or "amortization" in low:
                rate = re.search(r"(\d+(?:\.\d+)?)\s?%", text)
                years = re.search(r"(\d+)\s?(?:-| )?years?", low)
                return _call("calculate_amortization_schedule", {
                    "principal": _amount(text) or 300000, "interest_rate": float(rate.group(1)) if rate else 6.5,
                    "term_months": int(years.group(1)) * 12 if years else 360}, "Compute the payment.")
            product = "auto_new" if "auto" in low or "car" in low else "mortgage_30y_fixed"
            return _call("get_loan_product_rates", {"product_type": product}, "Fetch product rates.")
        if agent == "investments":
            if "invest" in low and _amount(text):
                return _call("execute_fund_subscription", {"acc_id": _first(r"INV-\d{6}", text, "INV-400001"),
                                                           "fund_id": _first(r"FND-[A-Z0-9]+", text, "FND-SPX500"),
                                                           "amount": _amount(text)}, "Subscribe to the fund.")
            if "inv-" in low:
                return _call("get_portfolio_valuation", {"investment_acc_id": _first(r"INV-\d{6}", text)},
                             "Value the portfolio.")
            return _call("fetch_cd_maturation_rates", {}, "Fetch CD rates.")
        if agent == "knowledge":
            z = _first(r"\b\d{5}\b", text)
            if z and ("atm" in low or "branch" in low or "nearest" in low):
                return _call("locate_nearest_atm_or_branch", {"zip_code_or_coords": z}, "Locate nearby branches.")
            return _call("query_knowledge_base", {"semantic_search_string": text[:400]}, "Search the policies.")
        return AIMessage(content="I can't help with that.")
    return respond


def offline_model_factory(role: str, agent: str | None = None) -> ScriptedChatModel:
    if role == "subagent" and agent:
        return ScriptedChatModel(responder=agent_responder(agent), model_name=f"offline-{agent}")
    return ScriptedChatModel(responder=supervisor_responder, model_name="offline-supervisor")
