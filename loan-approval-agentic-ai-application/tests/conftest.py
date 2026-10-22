from __future__ import annotations

import os

os.environ.update({"APP_ENV": "test", "LLM_CHAIN": "rules", "TRANSPORT_MODE": "inproc", "REDIS_URL": "", "POSTGRES_DSN": "",
                   "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "", "MAX_RETRIES": "3", "LOG_LEVEL": "WARNING"})

from typing import Any  # noqa: E402

import pytest  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from loanflow.common.config import reset_settings_cache  # noqa: E402
from loanflow.llm import client as llm_client  # noqa: E402
from loanflow.llm.client import LLMResult, LLMUnavailable, LLMUsage  # noqa: E402
from loanflow.memory import kv as kv_mod  # noqa: E402
from loanflow.memory import long_term  # noqa: E402
from loanflow.resilience import reset_breakers  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    reset_settings_cache()
    kv_mod.set_kv(kv_mod.InMemoryKV())
    long_term.set_long_term_memory(long_term.LongTermMemory(long_term.InMemoryVectorStore()))
    llm_client.set_llm(None)
    reset_breakers()
    yield
    reset_settings_cache()
    llm_client.set_llm(None)


class ScriptedLLM:
    """Deterministic stand-in for an LLM: returns queued JSON replies per call-name prefix."""

    def __init__(self, script: dict[str, list[dict[str, Any]]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[dict[str, str]] = []

    async def chat_json(self, *, system: str, user: str, schema: type[BaseModel], name: str, small: bool = False) -> LLMResult:
        self.calls.append({"name": name, "system": system, "user": user})
        for prefix, replies in self.script.items():
            if name.startswith(prefix) and replies:
                reply = replies.pop(0)
                return LLMResult(parsed=schema.model_validate(reply), raw=str(reply),
                                 usage=LLMUsage(provider="scripted", model="fake", prompt_tokens=100, completion_tokens=20))
        raise LLMUnavailable("script exhausted")


@pytest.fixture
def scripted_llm():
    def make(script):
        llm = ScriptedLLM(script)
        llm_client.set_llm(llm)  # type: ignore[arg-type]
        return llm
    return make


@pytest.fixture
def orchestrator():
    from loanflow.supervisor.graph import build_graph
    from loanflow.supervisor.runner import LoanOrchestrator

    return LoanOrchestrator(build_graph(checkpointer=MemorySaver()))
