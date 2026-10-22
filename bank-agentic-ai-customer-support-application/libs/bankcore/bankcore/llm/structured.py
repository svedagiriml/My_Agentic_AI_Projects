"""Model-agnostic structured output: JSON extraction + pydantic validation + repair loop.

Small free models don't always honour native JSON-mode/function-calling reliably, so we ask
for JSON, extract the first balanced object, validate with pydantic and, on failure, feed the
validation error back for a bounded number of repair attempts.
"""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from bankcore.errors import LLMOutputParseError
from bankcore.llm.factory import ResilientLLM

M = TypeVar("M", bound=BaseModel)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    if not text:
        raise ValueError("empty model output")
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object")


def parse_model(text: str, model_cls: type[M]) -> M:
    return model_cls.model_validate(extract_json(text))


async def invoke_structured(llm: ResilientLLM, messages: list[BaseMessage], model_cls: type[M], *,
                            retries: int = 2, config: dict | None = None) -> M:
    convo = list(messages)
    last_err = ""
    for _ in range(retries + 1):
        ai = await llm.ainvoke(convo, config=config)
        content = ai.content if isinstance(ai.content, str) else json.dumps(ai.content)
        try:
            return parse_model(content, model_cls)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last_err = str(e)[:800]
            convo = convo + [ai, HumanMessage(content=(
                "Your previous output was not valid. Error:\n" + last_err +
                "\nReturn ONLY one JSON object that satisfies the required schema. No prose."))]
    raise LLMOutputParseError(f"could not parse {model_cls.__name__}: {last_err}")
