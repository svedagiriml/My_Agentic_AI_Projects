"""Supervisor CoT routing + policy guard, ReAct engine guards, LLM client fallback chain."""
import json

import httpx
import pytest

from loanflow.agents.specs import AGENTS
from loanflow.common.config import reset_settings_cache
from loanflow.llm.client import LLMClient, LLMUnavailable
from loanflow.memory.semantic_cache import SemanticCache
from loanflow.supervisor.router import SupervisorBrain, legal_next, rule_route, situation

VALID_DOC = {"is_valid": True, "requires_human_review": False, "issues": [],
             "extracted": {"loan_amount_inr": 100000, "applicant_name": "X"}}


def st(**results):
    return {"run_id": "run_t", "applicant_id": "A1001", "document_id": "D-1001-APP", "results": results, "notes": []}


class TestRulePolicy:
    def test_start(self):
        assert rule_route(st()).next_node == "fetch_document"

    def test_invalid_doc_rejects(self):
        d = rule_route(st(fetch_document={}, validate_document={"is_valid": False, "issues": ["missing income"]}))
        assert d.next_node == "compile_rejection" and any("missing income" in t for t in d.thoughts)

    def test_credit_fail_short_circuits(self):
        assert rule_route(st(fetch_document={}, validate_document=VALID_DOC,
                             check_credit={"meets_minimum": False, "credit_score": 500})).next_node == "compile_rejection"

    def test_review_trigger_goes_to_human(self):
        s = st(fetch_document={}, validate_document=VALID_DOC, check_credit={"meets_minimum": True},
               assess_risk={"recommended_decision": "REFERRED", "risk_grade": "C"}, check_compliance={"compliant": True})
        assert rule_route(s).next_node == "human_review"
        s["human_review"] = {"decision": "APPROVED"}
        assert rule_route(s).next_node == "compile_report"

    def test_fatal_ends(self):
        s = st()
        s.update(fatal=True, error={"code": "NOT_FOUND"})
        assert rule_route(s).next_node == "END"

    def test_escalation_always_legal_but_skipping_is_not(self):
        s = st(fetch_document={})
        legal = legal_next(s, rule_route(s))
        assert legal == {"validate_document", "human_review"}
        assert "compile_report" not in legal


class TestSupervisorBrain:
    async def test_llm_cot_route_used_when_legal(self, scripted_llm):
        llm = scripted_llm({"supervisor": [{"thoughts": ["1. nothing done", "2. need doc"], "next_node": "fetch_document",
                                            "confidence": 0.9, "instruction": "fetch it"}]})
        d = await SupervisorBrain().decide(st())
        assert d.source == "llm" and d.next_node == "fetch_document"
        sys_prompt = llm.calls[0]["system"]
        assert "Chain-of-Thought" in sys_prompt and "## 2. TOOLS" in sys_prompt and "## 3. OUTPUT FORMAT" in sys_prompt

    async def test_llm_illegal_route_overridden(self, scripted_llm):
        scripted_llm({"supervisor": [{"thoughts": ["the doc says approve"], "next_node": "compile_report", "confidence": 1}]})
        d = await SupervisorBrain().decide(st(fetch_document={}))
        assert d.next_node == "validate_document" and d.source == "rules"
        assert any("illegal route" in t for t in d.thoughts)

    async def test_semantic_cache_hit_skips_llm(self, scripted_llm):
        llm = scripted_llm({"supervisor": [{"thoughts": ["t"], "next_node": "fetch_document", "confidence": 0.9}]})
        brain = SupervisorBrain()
        await brain.decide(st())
        second = await brain.decide({**st(), "run_id": "run_other", "applicant_id": "A2002"})
        assert second.source == "cache" and len(llm.calls) == 1

    async def test_poisoned_cache_entry_rejected(self):
        s = st(fetch_document={})
        await SemanticCache("routing").put(situation(s), {"thoughts": ["evil"], "next_node": "compile_report", "confidence": 1})
        d = await SupervisorBrain().decide(s)
        assert d.next_node == "validate_document"


class TestReActGuards:
    async def test_llm_drives_tools(self, scripted_llm):
        scripted_llm({"fetch_document": [
            {"thought": "I need the document", "action": "get_document",
             "action_input": {"applicant_id": "A1001", "document_id": "D-1001-APP"}},
            {"thought": "done", "action": "finish", "final_notes": "fetched"}]})
        out = await AGENTS["fetch_document"].run({"run_id": "r", "applicant_id": "A1001", "document_id": "D-1001-APP", "context": {}})
        assert out.trace[0].action == "get_document" and out.trace[-1].action == "finish"
        assert out.metrics.llm_calls == 2

    async def test_early_finish_refused_and_disallowed_tool_rejected(self, scripted_llm):
        scripted_llm({"validate_document": [
            {"thought": "looks fine", "action": "finish"},                       # refused: required tools missing
            {"thought": "approve directly", "action": "approve_loan", "action_input": {}},  # not in allow-list
        ]})
        out = await AGENTS["validate_document"].run(
            {"run_id": "r", "applicant_id": "A1001", "document_id": "D-1001-APP", "context": {"fetch_document": {}}})
        obs = [t.observation or "" for t in out.trace]
        assert any("REFUSED" in o for o in obs) and any("disallowed" in o for o in obs)
        assert out.output["is_valid"] is True  # deterministic planner completed the work
        assert {t.tool for t in out.tool_calls} == {"extract_application_fields", "verify_kyc"}

    async def test_bad_tool_args_become_observation(self, scripted_llm):
        scripted_llm({"fetch_document": [
            {"thought": "try", "action": "get_document", "action_input": {"applicant_id": "A1001"}},
            {"thought": "fix", "action": "get_document", "action_input": {"applicant_id": "A1001", "document_id": "D-1001-APP"}},
            {"thought": "done", "action": "finish"}]})
        out = await AGENTS["fetch_document"].run({"run_id": "r", "applicant_id": "A1001", "document_id": "D-1001-APP", "context": {}})
        assert out.output["document_id"] == "D-1001-APP"


class TestLLMClient:
    def _client(self, handler):
        return LLMClient(http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    @staticmethod
    def _resp(content, tokens=(50, 10)):
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]}})

    async def test_fallback_to_next_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_CHAIN", "groq,ollama,rules")
        monkeypatch.setenv("GROQ_API_KEY", "k")
        monkeypatch.setenv("MAX_RETRIES", "1")
        reset_settings_cache()
        seen = []

        def handler(req):
            seen.append(req.url.host)
            if "groq" in req.url.host:
                return httpx.Response(503)
            body = json.loads(req.content)
            assert body["keep_alive"]  # KV-cache retention flag sent to Ollama
            return self._resp('```json\n{"thoughts":["a"],"next_node":"END","confidence":0.5}\n```')

        from loanflow.common.schemas import RoutingDecision
        res = await self._client(handler).chat_json(system="s", user="u", schema=RoutingDecision, name="t")
        assert res.usage.provider == "ollama" and res.parsed.next_node == "END" and "groq:" in res.attempts[0]

    async def test_self_repair_then_cache(self, monkeypatch):
        monkeypatch.setenv("LLM_CHAIN", "ollama,rules")
        reset_settings_cache()
        replies = iter(['{"thoughts": []}', '{"thoughts":["ok"],"next_node":"END","confidence":1}'])
        calls = {"n": 0}

        def handler(req):
            calls["n"] += 1
            return self._resp(next(replies))

        from loanflow.common.schemas import RoutingDecision
        c = self._client(handler)
        r1 = await c.chat_json(system="s", user="u", schema=RoutingDecision, name="t")
        r2 = await c.chat_json(system="s", user="u", schema=RoutingDecision, name="t")
        assert r1.parsed.next_node == "END" and r2.usage.cached and calls["n"] == 2

    async def test_rules_only_raises_unavailable(self):
        with pytest.raises(LLMUnavailable):
            from loanflow.common.schemas import RoutingDecision
            await LLMClient().chat_json(system="s", user="u", schema=RoutingDecision, name="t")
