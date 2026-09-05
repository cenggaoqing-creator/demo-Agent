from __future__ import annotations

import json
from datetime import date
from typing import Any


class ContextManager:
    def __init__(self, limit: int = 12000, keep_messages: int = 8) -> None:
        self.limit = limit
        self.keep_messages = keep_messages

    def compact(self, session: dict[str, Any]) -> bool:
        messages = session["messages"]
        encoded = json.dumps(messages, ensure_ascii=False)
        if len(encoded) <= self.limit:
            return False
        old = messages[:-self.keep_messages]
        facts = []
        for msg in old:
            content = str(msg.get("content", "")).replace("\n", " ")
            if content:
                facts.append(f"{msg.get('role', 'unknown')}: {content[:240]}")
        prior = session.get("summary", "")
        session["summary"] = (prior + "\n" + "\n".join(facts))[-4000:]
        session["messages"] = messages[-self.keep_messages:]
        return True

    def build(self, session: dict[str, Any], tool_schemas: list[dict[str, Any]], memories: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        today = date.today().isoformat()
        instruction = (
            "你是一个最小可用 Agent。只能通过提供的工具完成外部操作。"
            "用户一次请求多个外部信息或操作时，必须逐项调用所有相关工具；没有实际执行的工具不能声称已完成，也不能用稍后、我将查询等话术代替结果。"
            "必须只输出一个 JSON 对象，严格使用以下格式之一："
            '{"type":"final","reasoning_summary":"简短原因","answer":"给用户的答案"}、'
            '{"type":"tool_call","reasoning_summary":"简短原因","tool_call":{"name":"工具名","arguments":{}}}，或 '
            '{"type":"tool_calls","reasoning_summary":"简短原因","tool_calls":[{"name":"工具名","arguments":{}}]}。'
            "同一轮多个操作必须使用 tool_calls 数组，绝不能重复 tool_call 字段。"
            "收入和支出使用 expense_tracker.add，必须提供 entry_type=income 或 expense；查询余额变化使用 expense_tracker.summary。"
            "数学表达式使用 calculator；资料搜索使用 search。"
            f"当前系统日期是 {today}；用户说‘今天’且需要按日查询时，使用 expense_date={today}。"
            "不要输出 Markdown、前后解释或其他字段。"
        )
        system = {"role": "system", "content": instruction}
        tools = {"role": "system", "content": "可用工具 Schema：" + json.dumps(tool_schemas, ensure_ascii=False)}
        context = [system, tools]
        if session.get("summary"):
            context.append({"role": "system", "content": "历史摘要：" + session["summary"]})
        if memories:
            context.append({"role": "system", "content": "用户明确要求记住的长期记忆：" + json.dumps([item["content"] for item in memories], ensure_ascii=False)})
        context.extend(session["messages"])
        return context
