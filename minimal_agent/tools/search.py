from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from .base import ToolSpec


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=120, description="要搜索的关键词或问题")
    top_k: int = Field(default=3, ge=1, le=5, description="最多返回的结果数")


_MOCK_DOCUMENTS = (
    {
        "id": "agent-runtime",
        "title": "Minimal Agent Runtime",
        "url": "https://docs.example.local/agent-runtime",
        "snippet": "Agent Runtime 负责工具调用循环、状态保存、错误处理和 trace。",
    },
    {
        "id": "tool-schema",
        "title": "Tool Schema Guide",
        "url": "https://docs.example.local/tool-schema",
        "snippet": "工具通过名称、描述、JSON Schema 和副作用元数据向模型声明能力边界。",
    },
    {
        "id": "session-memory",
        "title": "Session and Memory",
        "url": "https://docs.example.local/session-memory",
        "snippet": "Session 隔离窗口上下文，长期 Memory 只保存用户明确授权的偏好。",
    },
    {
        "id": "weather-mock",
        "title": "Weather Mock Data",
        "url": "https://docs.example.local/weather",
        "snippet": "天气工具使用固定 mock 数据，便于离线测试和可重复演示。",
    },
)


def _terms(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", query.lower())


def _handle(args: SearchArgs, *, session_state: dict[str, Any]) -> dict[str, Any]:
    query = args.query.strip()
    normalized_query = query.lower()
    terms = _terms(query)
    ranked: list[tuple[int, dict[str, Any]]] = []
    for document in _MOCK_DOCUMENTS:
        searchable = " ".join((document["title"], document["snippet"])).lower()
        score = 10 if normalized_query in searchable else 0
        score += sum(1 for term in terms if term in searchable)
        if score:
            ranked.append((score, document))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    results = [
        {
            "id": document["id"],
            "title": document["title"],
            "url": document["url"],
            "snippet": document["snippet"],
            "score": score,
        }
        for score, document in ranked[: args.top_k]
    ]
    return {"query": query, "results": results, "count": len(results), "source": "local_mock"}


def search_tool() -> ToolSpec:
    return ToolSpec(
        "search",
        "在内置 mock 文档库中搜索 Agent、工具、Session、Memory 和天气相关资料，不访问外部网络",
        SearchArgs,
        _handle,
        side_effect="read",
    )
