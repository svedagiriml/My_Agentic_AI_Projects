"""Deterministic scripted chat model for unit/integration tests (no network, no GPU)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

Responder = Callable[[list[BaseMessage]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """Returns responses from ``script`` in order, or computes them with ``responder``."""

    script: list[AIMessage] = Field(default_factory=list)
    responder: Responder | None = None
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    model_name: str = "scripted-fake"

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":  # type: ignore[override]
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        if self.responder is not None:
            msg = self.responder(messages)
        elif self.script:
            msg = self.script.pop(0)
        else:
            msg = AIMessage(content="(no scripted response)")
        msg = msg.model_copy()
        msg.usage_metadata = {"input_tokens": sum(len(str(m.content)) // 4 for m in messages),
                              "output_tokens": len(str(msg.content)) // 4 + 1,
                              "total_tokens": 0}
        msg.response_metadata = {"model_name": self.model_name}
        return ChatResult(generations=[ChatGeneration(message=msg)])
