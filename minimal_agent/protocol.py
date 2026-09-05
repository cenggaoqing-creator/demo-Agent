"""LLM decision protocol and tolerant JSON parsing."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One validated tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any]


@dataclass
class Decision:
    kind: str
    reasoning_summary: str = ""
    answer: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class ProtocolError(ValueError):
    pass


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise ProtocolError(f"JSON 对象包含重复字段: {key}；多个工具调用请使用 tool_calls 数组")
        data[key] = value
    return data


def _parse_tool_call(value: Any, field_name: str) -> ToolCall:
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"].strip():
        raise ProtocolError(f"{field_name} 缺少工具名称")
    arguments = value.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ProtocolError(f"{field_name}.arguments 必须是对象")
    return ToolCall(value["name"].strip(), arguments)


def _tool_decision(kind: str, reasoning_summary: str, calls: list[ToolCall]) -> Decision:
    if not calls:
        raise ProtocolError("tool_calls 不能为空")
    # Keep the first-call fields for code written against the original single-call protocol.
    first = calls[0]
    return Decision(kind, reasoning_summary, tool_name=first.name, arguments=first.arguments, tool_calls=calls)


def parse_decision(raw: str) -> Decision:
    """Parse final, one tool call, or an ordered batch of tool calls."""
    text = (raw or "").strip()
    if not text:
        raise ProtocolError("LLM 返回为空")
    candidates = [text]
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if match:
        candidates.insert(0, match.group(1))
    data = None
    for candidate in candidates:
        try:
            data = json.loads(candidate, object_pairs_hook=_no_duplicate_keys)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        raise ProtocolError("LLM 输出不是 JSON 对象")
    kind = data.get("type") or data.get("kind")
    # Some OpenAI-compatible models omit the envelope and return {"final": ...}
    # or {"tool_call": ...}; normalize those common variants.
    if kind is None and isinstance(data.get("final"), str):
        return Decision("final", str(data.get("reasoning_summary", "")), data["final"].strip())
    if kind is None and isinstance(data.get("answer"), str):
        return Decision("final", str(data.get("reasoning_summary", "")), data["answer"].strip())
    if kind is None and isinstance(data.get("tool_call"), dict):
        kind = "tool_call"
    if kind is None and isinstance(data.get("tool_calls"), list):
        kind = "tool_calls"
    if kind == "final":
        answer = data.get("answer") or data.get("final")
        if not isinstance(answer, str) or not answer.strip():
            raise ProtocolError("final 缺少 answer")
        return Decision("final", str(data.get("reasoning_summary", "")), answer.strip())
    if kind == "tool_call":
        return _tool_decision("tool_call", str(data.get("reasoning_summary", "")), [_parse_tool_call(data.get("tool_call"), "tool_call")])
    if kind == "tool_calls":
        calls = data.get("tool_calls")
        if not isinstance(calls, list):
            raise ProtocolError("tool_calls 必须是数组")
        return _tool_decision("tool_calls", str(data.get("reasoning_summary", "")), [_parse_tool_call(call, f"tool_calls[{index}]") for index, call in enumerate(calls)])
    raise ProtocolError(f"未知决策类型: {kind!r}")
