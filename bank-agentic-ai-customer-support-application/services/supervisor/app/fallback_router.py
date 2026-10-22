"""Deterministic keyword router — graceful degradation when every LLM provider is down
(or its circuit is open). Splits multi-intent messages on conjunctions and routes each
clause by keyword scores. Accuracy is lower than the CoT planner, but the bank stays up."""
from __future__ import annotations

import re

from bankcore.schemas.common import AgentName
from bankcore.schemas.messages import PlannedTask, SupervisorPlan

KEYWORDS: dict[AgentName, list[str]] = {
    AgentName.FRAUD: ["fraud", "unauthori", "don't recognize", "dont recognize", "don't recognise", "not me",
                      "hacked", "suspicious", "someone logged", "stolen identity", "scam", "takeover",
                      "unknown login", "lock my profile"],
    AgentName.CARDS: ["card", "lost", "stolen", "freeze", "unfreeze", "replacement", "atm limit", "withdrawal limit"],
    AgentName.PAYMENTS: ["transfer", "send money", "move", "pay bill", "bill pay", "schedule payment", "routing number valid",
                         "verify routing", "exchange rate", "wire", "fx", "remit"],
    AgentName.ACCOUNT: ["balance", "transactions", "statement", "spent", "spending", "history", "how much do i have", "how much money", "savings account",
                          "checking account"],
    AgentName.LOANS: ["loan", "mortgage", "amortization", "amortisation", "emi", "monthly payment", "heloc", "refinanc",
                      "application status", "app-"],
    AgentName.INVESTMENTS: ["cd rate", "certificate of deposit", "portfolio", "ira", "mutual fund", "invest",
                            "fund", "inv-"],
    AgentName.KNOWLEDGE: ["fee", "hours", "branch", "atm near", "nearest", "routing number", "swift", "policy",
                          "overdraft", "zelle", "privacy", "holiday", "wire cut-off", "dispute", "investigation"],
}
_SPLIT = re.compile(r"\b(?:and also|also|and then|oh by the way|by the way|btw|plus|additionally)\b|[;?]\s+|\.\s+", re.I)


def _score(clause: str) -> tuple[AgentName | None, int]:
    low = clause.lower()
    best, best_score = None, 0
    for agent, kws in KEYWORDS.items():
        sc = sum(2 if " " in k else 1 for k in kws if k in low)
        if sc > best_score:
            best, best_score = agent, sc
    return best, best_score


def keyword_plan(message: str) -> SupervisorPlan:
    clauses = [c.strip(" ,.") for c in _SPLIT.split(message) if c and len(c.strip()) > 3] or [message]
    tasks: list[PlannedTask] = []
    seen: set[AgentName] = set()
    for clause in clauses:
        agent, sc = _score(clause)
        if agent is None or sc == 0 or agent in seen:
            continue
        seen.add(agent)
        tasks.append(PlannedTask(task_id=f"t{len(tasks) + 1}", agent=agent, instruction=clause[:1000],
                                 priority=1 if agent == AgentName.FRAUD else 5))
    if not tasks:
        agent, _ = _score(message)
        tasks = [PlannedTask(task_id="t1", agent=agent or AgentName.KNOWLEDGE, instruction=message[:1000])]
    # fraud first
    tasks.sort(key=lambda t: 0 if t.agent == AgentName.FRAUD else 1)
    tasks = [t.model_copy(update={"task_id": f"t{i + 1}"}) for i, t in enumerate(tasks)]
    return SupervisorPlan(reasoning=["Degraded mode: keyword routing (LLM planner unavailable)."],
                          intents=[t.agent.value for t in tasks], tasks=tasks, confidence=0.4)
