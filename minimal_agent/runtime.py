from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from .context import ContextManager
from .errors import LLMError, ToolNotFoundError, error_info
from .memory import MemoryStore
from .protocol import Decision, ProtocolError, parse_decision
from .session import SessionStore
from .tools import default_registry
from .tools.base import ToolRegistry
from .tracing import Trace


class LLM(Protocol):
    def complete(self, messages: list[dict[str, Any]]) -> str: ...


@dataclass
class AgentResult:
    answer: str
    session_id: str
    trace_id: str
    loops: int
    tools_called: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"
    llm_calls: int = 0
    total_latency_ms: float = 0.0
    memories_recalled: list[str] = field(default_factory=list)


class AgentRuntime:
    def __init__(self, llm: LLM, *, store: SessionStore | None = None, registry: ToolRegistry | None = None, context: ContextManager | None = None, trace_dir: str | None = None, max_loops: int | None = None, memory_store: MemoryStore | None = None, confirm_writes: bool | None = None, llm_retries: int | None = None, retry_backoff: float | None = None) -> None:
        self.llm = llm
        self.store = store or SessionStore(os.getenv("AGENT_DB_PATH", "runtime/minimal_agent.db"))
        self.registry = registry or default_registry()
        self.context = context or ContextManager(int(os.getenv("AGENT_CONTEXT_LIMIT", "12000")))
        self.trace_dir = trace_dir or os.getenv("AGENT_TRACE_DIR", "runtime/traces")
        self.max_loops = int(os.getenv("AGENT_MAX_LOOPS", "8")) if max_loops is None else max_loops
        self.memory_store = memory_store or MemoryStore(os.getenv("AGENT_DB_PATH", "runtime/minimal_agent.db"))
        self.confirm_writes = (os.getenv("AGENT_CONFIRM_WRITES", "false").lower() == "true") if confirm_writes is None else confirm_writes
        self.llm_retries = int(os.getenv("AGENT_LLM_RETRIES", "2")) if llm_retries is None else llm_retries
        self.retry_backoff = float(os.getenv("AGENT_RETRY_BACKOFF", "0.1")) if retry_backoff is None else retry_backoff

    @staticmethod
    def _explicit_memory(text: str) -> str | None:
        match = re.search(r"(?:请)?记住我(?:以后)?[：:\s]*(.+)$", text.strip())
        return match.group(1).strip() if match else None

    @staticmethod
    def _required_tools(text: str) -> list[str]:
        """A small completion guard for explicit external actions in Chinese requests."""
        required: list[str] = []
        if re.search(r"天气|气温|温度|下雨", text) and "weather" not in required:
            required.append("weather")
        if re.search(r"支出|收入|余额|收支|花了|记一笔|记账|报销|账单|费用", text) and "expense_tracker" not in required:
            required.append("expense_tracker")
        if re.search(r"计算|算(?:一下|一算|出)?|等于|\d+\s*(?:\+|\*|/|%|加|乘|除)\s*\d+", text) and "calculator" not in required:
            required.append("calculator")
        if re.search(r"搜索|检索|查找(?:资料|信息|文档|内容)?", text) and "search" not in required:
            required.append("search")
        return required

    @classmethod
    def _required_tool_counts(cls, text: str) -> dict[str, int]:
        """Derive the minimum successful calls needed for clear multi-action requests."""
        counts = {name: 1 for name in cls._required_tools(text)}
        records_income_and_expense = (
            bool(re.search(r"记录|记一笔|记账|新增|添加", text))
            and "收入" in text
            and bool(re.search(r"支出|花了|费用", text))
        )
        if records_income_and_expense:
            counts["expense_tracker"] = 2
        return counts

    def _complete_with_retry(self, messages: list[dict[str, Any]], trace: Trace, loop: int) -> str:
        last_error: Exception | None = None
        for attempt in range(self.llm_retries + 1):
            started = time.perf_counter()
            try:
                self._llm_calls += 1
                raw = self.llm.complete(messages)
                trace.record("llm_response", loop=loop, attempt=attempt + 1, latency_ms=round((time.perf_counter() - started) * 1000, 2), raw=raw[:4000])
                return raw
            except Exception as exc:
                last_error = exc
                trace.record("llm_failed", loop=loop, attempt=attempt + 1, latency_ms=round((time.perf_counter() - started) * 1000, 2), **error_info(exc))
                if attempt < self.llm_retries and self.retry_backoff > 0:
                    time.sleep(self.retry_backoff * (2**attempt))
        raise LLMError(f"LLM 调用失败: {last_error}", recoverable=False) from last_error

    def run(self, user_input: str, *, session_id: str | None = None, user_id: str | None = None) -> AgentResult:
        if not user_input.strip():
            raise ValueError("用户输入不能为空")
        sid = session_id or uuid.uuid4().hex
        session = self.store.load(sid, user_id)
        trace = Trace(self.trace_dir, sid)
        started_request = time.perf_counter()
        self._llm_calls = 0
        trace.record("request_started", input=user_input)
        memories = self.memory_store.recall(user_id or "", user_input)
        explicit_memory = self._explicit_memory(user_input)
        if explicit_memory and user_id:
            self.memory_store.remember(user_id, explicit_memory)
            memories = self.memory_store.recall(user_id, user_input)
            if not any(item.get("content") == explicit_memory for item in memories):
                memories.append({"content": explicit_memory})
            trace.record("memory_saved", content=explicit_memory)
        elif explicit_memory:
            trace.record("memory_skipped", reason="缺少 user_id", content=explicit_memory)
        if memories:
            trace.record("memory_recalled", count=len(memories), memories=[item["content"] for item in memories])
        session["messages"].append({"role": "user", "content": user_input})
        tools_called: list[str] = []
        required_tools = self._required_tools(user_input)
        required_tool_counts = self._required_tool_counts(user_input)
        trace.record("completion_requirements", required_tools=required_tools, required_tool_counts=required_tool_counts)
        tools_failed: set[str] = set()
        last_missing_tools: tuple[str, ...] | None = None
        stalled_final_count = 0
        for loop in range(1, self.max_loops + 1):
            compacted = self.context.compact(session)
            if compacted:
                trace.record("context_compacted", loop=loop)
            messages = self.context.build(session, self.registry.schemas(), memories)
            try:
                raw = self._complete_with_retry(messages, trace, loop)
                try:
                    decision = parse_decision(raw)
                except ProtocolError as first_error:
                    trace.record("protocol_retry", loop=loop, error=str(first_error))
                    repair_messages = messages + [{"role": "system", "content": "上一次输出格式不合法。请只输出符合协议的 JSON，不要 Markdown 或解释。"}]
                    raw = self._complete_with_retry(repair_messages, trace, loop)
                    decision = parse_decision(raw)
            except Exception as exc:
                trace.record("request_failed", loop=loop, **error_info(exc))
                session["messages"].append({"role": "assistant", "content": f"处理失败：{exc}"})
                self.store.save(session)
                return AgentResult(f"处理失败：{exc}", sid, trace.trace_id, loop, tools_called, trace.events, "failed", self._llm_calls, round((time.perf_counter() - started_request) * 1000, 2), [item["content"] for item in memories])
            trace.record("parsed_decision", loop=loop, kind=decision.kind, reasoning_summary=decision.reasoning_summary, tool_count=len(decision.tool_calls))
            if decision.kind == "final":
                missing_tools = [
                    name
                    for name, required_count in required_tool_counts.items()
                    if tools_called.count(name) < required_count and name not in tools_failed
                ]
                if missing_tools:
                    missing_requirements = {name: required_tool_counts[name] - tools_called.count(name) for name in missing_tools}
                    trace.record("completion_guard", loop=loop, missing_tools=missing_tools, missing_requirements=missing_requirements)
                    missing_key = tuple(f"{name}:{missing_requirements[name]}" for name in missing_tools)
                    stalled_final_count = stalled_final_count + 1 if missing_key == last_missing_tools else 1
                    last_missing_tools = missing_key
                    if stalled_final_count >= 2:
                        answer = "Agent 未产生满足请求所需的工具调用，已停止无效重试。请补充必要信息后重试。"
                        trace.record("completion_stalled", loop=loop, missing_tools=missing_tools, missing_requirements=missing_requirements, repeated_final_count=stalled_final_count)
                        session["messages"].append({"role": "assistant", "content": answer})
                        self.store.save(session)
                        return AgentResult(answer, sid, trace.trace_id, loop, tools_called, trace.events, "stalled", self._llm_calls, round((time.perf_counter() - started_request) * 1000, 2), [item["content"] for item in memories])
                    required_text = ", ".join(f"{name}（还需 {count} 次）" for name, count in missing_requirements.items())
                    session["messages"].append({"role": "system", "content": "用户请求尚未完成，必须先成功调用这些工具后才能回答：" + required_text})
                    self.store.save(session)
                    continue
                session["messages"].append({"role": "assistant", "content": decision.answer})
                session["turn_count"] += 1
                trace.record("final_answer", loop=loop, answer=decision.answer, latency_ms=round((time.perf_counter() - started_request) * 1000, 2))
                self.store.save(session)
                return AgentResult(decision.answer, sid, trace.trace_id, loop, tools_called, trace.events, "completed", self._llm_calls, round((time.perf_counter() - started_request) * 1000, 2), [item["content"] for item in memories])
            last_missing_tools = None
            stalled_final_count = 0
            requested_calls = decision.tool_calls
            assistant_tool_calls = [
                {
                    "id": f"call_{trace.trace_id[:10]}_{loop}_{index}",
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                }
                for index, call in enumerate(requested_calls, start=1)
            ]
            session["messages"].append({
                "role": "assistant",
                "content": json.dumps({"tool_calls": [{"name": call.name, "arguments": call.arguments} for call in requested_calls]}, ensure_ascii=False),
                "tool_calls": assistant_tool_calls,
            })
            for index, requested_call in enumerate(requested_calls, start=1):
                name = requested_call.name
                call_id = f"call_{trace.trace_id[:10]}_{loop}_{index}"
                try:
                    try:
                        tool = self.registry.get(name)
                    except KeyError as exc:
                        raise ToolNotFoundError(str(exc)) from exc
                    validated = tool.validate(requested_call.arguments)
                    normalized_arguments = validated.model_dump(exclude_none=True)
                    if self.confirm_writes and tool.needs_confirmation(validated) and not normalized_arguments.get("confirmed", False):
                        trace.record("confirmation_required", loop=loop, tool=name, side_effect=tool.side_effect)
                        answer = f"工具 {name} 将执行写操作，请在确认后重试。"
                        session["messages"].append({"role": "assistant", "content": answer})
                        self.store.save(session)
                        return AgentResult(answer, sid, trace.trace_id, loop, tools_called, trace.events, "confirmation_required", self._llm_calls, round((time.perf_counter() - started_request) * 1000, 2), [item["content"] for item in memories])
                    trace.record("tool_started", loop=loop, tool=name, call_index=index, arguments=normalized_arguments, side_effect=tool.side_effect)
                    operation_id = normalized_arguments.get("operation_id") if tool.side_effect == "write" else None
                    operations = session["state"].setdefault("_operations", {})
                    if operation_id and operation_id in operations:
                        result = operations[operation_id]
                        trace.record("tool_idempotent_replay", loop=loop, tool=name, call_index=index, operation_id=operation_id)
                    else:
                        result = tool.invoke(normalized_arguments, session_state=session["state"])
                        if operation_id:
                            operations[operation_id] = result
                    tools_called.append(name)
                    session["messages"].append({"role": "tool", "name": name, "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False, default=str)})
                    trace.record("tool_finished", loop=loop, tool=name, call_index=index, result=result, operation_id=operation_id)
                    self.store.save(session)
                except Exception as exc:
                    tools_failed.add(name)
                    trace.record("tool_failed", loop=loop, tool=name, call_index=index, **error_info(exc))
                    session["messages"].append({"role": "tool", "name": name, "tool_call_id": call_id, "content": json.dumps({"error": str(exc)}, ensure_ascii=False)})
                    self.store.save(session)
        trace.record("max_loops_reached", max_loops=self.max_loops)
        answer = "已达到最大推理轮次，暂时无法完成请求。"
        session["messages"].append({"role": "assistant", "content": answer})
        self.store.save(session)
        return AgentResult(answer, sid, trace.trace_id, self.max_loops, tools_called, trace.events, "max_loops", self._llm_calls, round((time.perf_counter() - started_request) * 1000, 2), [item["content"] for item in memories])
