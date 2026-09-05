import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from minimal_agent.context import ContextManager
from minimal_agent.errors import ToolExecutionError
from minimal_agent.memory import MemoryStore
from minimal_agent.protocol import ProtocolError, parse_decision
from minimal_agent.runtime import AgentRuntime
from minimal_agent.session import SessionStore
from minimal_agent.tools import default_registry
from minimal_agent.tools.base import ToolRegistry
from minimal_agent.tools.calculator import calculator_tool
from minimal_agent.tools.search import search_tool


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return self.responses.pop(0)


class FlakyLLM:
    def __init__(self, final_response):
        self.failed = False
        self.final_response = final_response

    def complete(self, messages):
        if not self.failed:
            self.failed = True
            raise TimeoutError("模拟超时")
        return self.final_response


class MinimalAgentTests(unittest.TestCase):
    def make_runtime(self, llm, temp, **kwargs):
        return AgentRuntime(
            llm,
            store=SessionStore(str(Path(temp) / "db.sqlite")),
            trace_dir=str(Path(temp) / "traces"),
            context=ContextManager(500, 2),
            max_loops=kwargs.pop("max_loops", 3),
            confirm_writes=kwargs.pop("confirm_writes", False),
            **kwargs,
        )

    def test_direct_answer(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.make_runtime(ScriptedLLM('{"type":"final","answer":"你好"}'), d).run("你好", session_id="s1")
            self.assertEqual(result.answer, "你好")
            self.assertEqual(result.tools_called, [])

    def test_weather_tool_chain(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"weather","arguments":{"city":"上海"}}}',
                '{"type":"final","answer":"上海晴，26度"}',
            )
            result = self.make_runtime(llm, d).run("查询上海天气", session_id="weather")
            self.assertEqual(result.tools_called, ["weather"])
            self.assertIn("上海", llm.messages[1][-1]["content"])

    def test_weather_unknown_city_is_safe(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"weather","arguments":{"city":"深圳"}}}',
                '{"type":"final","answer":"暂无数据"}',
            )
            result = self.make_runtime(llm, d).run("查深圳天气", session_id="weather-unknown")
            self.assertEqual(result.status, "completed")
            self.assertIn("未知", llm.messages[1][-1]["content"])

    def test_expense_add_persists_state(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"add","entry_type":"expense","amount":35.5,"category":"交通","note":"地铁","expense_date":"2026-09-04"}}}',
                '{"type":"final","answer":"已记录"}',
            )
            result = self.make_runtime(llm, d).run("记录交通支出 35.5 元", session_id="expense")
            self.assertEqual(result.tools_called, ["expense_tracker"])
            state = SessionStore(str(Path(d) / "db.sqlite")).load("expense")["state"]
            self.assertEqual(state["expenses"][0]["amount"], 35.5)
            self.assertEqual(state["expenses"][0]["category"], "交通")
            self.assertEqual(state["expenses"][0]["entry_type"], "expense")

    def test_expense_list_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "db.sqlite")
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"add","entry_type":"expense","amount":10,"category":"餐饮"}}}',
                '{"type":"final","answer":"已记账"}',
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"summary"}}}',
                '{"type":"final","answer":"合计10元"}',
            )
            runtime = AgentRuntime(llm, store=SessionStore(path), trace_dir=str(Path(d) / "traces"), max_loops=3, confirm_writes=False)
            runtime.run("记一笔餐饮支出", session_id="expense-summary")
            result = runtime.run("汇总支出", session_id="expense-summary")
            self.assertEqual(result.tools_called, ["expense_tracker"])
            self.assertIn('"expense_total": 10.0', llm.messages[-1][-1]["content"])

    def test_expense_remove(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"add","entry_type":"expense","amount":8,"category":"餐饮","expense_id":"exp-a"}}}',
                '{"type":"final","answer":"已记录"}',
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"remove","expense_id":"exp-a"}}}',
                '{"type":"final","answer":"已删除"}',
            )
            runtime = self.make_runtime(llm, d)
            runtime.run("记一笔支出", session_id="expense-remove")
            result = runtime.run("删除这笔支出", session_id="expense-remove")
            self.assertEqual(result.status, "completed")
            self.assertEqual(SessionStore(str(Path(d) / "db.sqlite")).load("expense-remove")["state"]["expenses"], [])

    def test_calculator_tool_chain(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"calculator","arguments":{"expression":"(12 + 8) * 3 / 2"}}}',
                '{"type":"final","answer":"结果是 30"}',
            )
            result = self.make_runtime(llm, d).run("计算 (12 + 8) * 3 / 2", session_id="calculator")
            self.assertEqual(result.tools_called, ["calculator"])
            self.assertIn('"result": 30.0', llm.messages[1][-1]["content"])

    def test_calculator_respects_precedence_and_parentheses(self):
        result = calculator_tool().invoke({"expression": "2 + 3 * (4 - 1)"}, session_state={})
        self.assertEqual(result["result"], 11)

    def test_calculator_handles_decimals(self):
        result = calculator_tool().invoke({"expression": "7.5 / 2"}, session_state={})
        self.assertEqual(result["result"], 3.75)

    def test_calculator_rejects_arbitrary_code(self):
        with self.assertRaises(ToolExecutionError):
            calculator_tool().invoke({"expression": "__import__('os').system('whoami')"}, session_state={})

    def test_search_tool_chain(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool_call": {
                            "name": "search",
                            "arguments": {"query": "Agent Runtime", "top_k": 2},
                        },
                    },
                    ensure_ascii=False,
                ),
                '{"type":"final","answer":"找到了 Agent Runtime 文档"}',
            )
            result = self.make_runtime(llm, d).run("搜索 Agent Runtime 资料", session_id="search")
            self.assertEqual(result.tools_called, ["search"])
            tool_result = json.loads(llm.messages[1][-1]["content"])
            self.assertEqual(tool_result["results"][0]["id"], "agent-runtime")
            self.assertEqual(tool_result["source"], "local_mock")

    def test_search_returns_empty_results_safely(self):
        result = search_tool().invoke({"query": "zzzz_unmatched"}, session_state={})
        self.assertEqual(result["results"], [])
        self.assertEqual(result["count"], 0)

    def test_registry_contains_exactly_four_new_tools(self):
        names = {schema["name"] for schema in default_registry().schemas()}
        self.assertEqual(names, {"calculator", "search", "weather", "expense_tracker"})

    def test_required_tools_identify_expense_calculation_and_search(self):
        self.assertEqual(AgentRuntime._required_tools("记录 2026-09-04 的交通支出"), ["expense_tracker"])
        self.assertEqual(AgentRuntime._required_tools("计算 12 * 3，并搜索 Agent Runtime 资料"), ["calculator", "search"])

    def test_required_tools_treat_today_balance_as_expense_query(self):
        self.assertEqual(AgentRuntime._required_tools("今天余额变化"), ["expense_tracker"])

    def test_schema_contains_new_tool_fields(self):
        registry = default_registry()
        self.assertIn("amount", registry.get("expense_tracker").parameters["properties"])
        self.assertIn("expression", registry.get("calculator").parameters["properties"])
        self.assertIn("query", registry.get("search").parameters["properties"])

    def test_duplicate_tool_registration_rejected(self):
        registry = ToolRegistry()
        registry.register(calculator_tool())
        with self.assertRaises(ValueError):
            registry.register(calculator_tool())

    def test_session_isolation_for_expenses(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"add","entry_type":"expense","amount":10,"category":"A"}}}',
                '{"type":"final","answer":"一号"}',
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"list"}}}',
                '{"type":"final","answer":"二号为空"}',
            )
            runtime = self.make_runtime(llm, d)
            runtime.run("记账", session_id="window-1", user_id="user-a")
            runtime.run("查看账单", session_id="window-2", user_id="user-a")
            store = SessionStore(str(Path(d) / "db.sqlite"))
            self.assertEqual(len(store.load("window-1")["state"]["expenses"]), 1)
            self.assertEqual(store.load("window-2")["state"].get("expenses", []), [])

    def test_follow_up_can_use_previous_tool_result(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"weather","arguments":{"city":"上海"}}}',
                '{"type":"final","answer":"上海晴，26度"}',
                '{"type":"tool_call","tool_call":{"name":"calculator","arguments":{"expression":"26 + 4"}}}',
                '{"type":"final","answer":"30度"}',
            )
            runtime = self.make_runtime(llm, d)
            runtime.context = ContextManager(10000)
            runtime.run("查询上海天气", session_id="follow-up")
            runtime.run("把刚才的温度加 4", session_id="follow-up")
            self.assertTrue(any(m["role"] == "tool" and "26" in m["content"] for m in llm.messages[-2]))

    def test_completion_guard_requires_all_explicit_tools(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"final","answer":"都完成了"}',
                '{"type":"tool_call","tool_call":{"name":"weather","arguments":{"city":"上海"}}}',
                '{"type":"tool_call","tool_call":{"name":"calculator","arguments":{"expression":"2 + 3"}}}',
                '{"type":"tool_call","tool_call":{"name":"search","arguments":{"query":"Agent Runtime"}}}',
                '{"type":"final","answer":"天气、计算和搜索完成"}',
            )
            result = self.make_runtime(llm, d, max_loops=6).run(
                "查上海天气，计算 2 + 3，并搜索 Agent Runtime 资料",
                session_id="guard",
            )
            self.assertEqual(result.tools_called, ["weather", "calculator", "search"])
            self.assertTrue(any(e["event"] == "completion_guard" for e in result.events))

    def test_invalid_tool_arguments_are_handled(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"calculator","arguments":{"expression":"1 / 0"}}}',
                '{"type":"final","answer":"参数有误"}',
            )
            result = self.make_runtime(llm, d).run("计算 1 / 0", session_id="bad-args")
            self.assertEqual(result.status, "completed")
            self.assertTrue(any(e["event"] == "tool_failed" for e in result.events))

    def test_unknown_tool_is_returned_to_llm(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"missing_tool","arguments":{}}}',
                '{"type":"final","answer":"没有这个工具"}',
            )
            result = self.make_runtime(llm, d).run("调用不存在工具", session_id="unknown")
            self.assertEqual(result.answer, "没有这个工具")

    def test_malformed_output_returns_error(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.make_runtime(ScriptedLLM("not-json"), d).run("x", session_id="bad")
            self.assertIn("处理失败", result.answer)
            self.assertTrue(any(e["event"] == "request_failed" for e in result.events))

    def test_protocol_variants(self):
        self.assertEqual(parse_decision('{"final":"你好"}').answer, "你好")
        with self.assertRaises(ProtocolError):
            parse_decision('{"type":"unknown"}')

    def test_protocol_accepts_ordered_multiple_tool_calls(self):
        decision = parse_decision(
            '{"type":"tool_calls","tool_calls":[{"name":"weather","arguments":{"city":"上海"}},{"name":"calculator","arguments":{"expression":"2 + 3"}}]}'
        )
        self.assertEqual(decision.kind, "tool_calls")
        self.assertEqual([call.name for call in decision.tool_calls], ["weather", "calculator"])

    def test_protocol_rejects_duplicate_tool_call_fields(self):
        with self.assertRaises(ProtocolError):
            parse_decision(
                '{"tool_call":{"name":"weather","arguments":{"city":"上海"}},"tool_call":{"name":"weather","arguments":{"city":"北京"}}}'
            )

    def test_protocol_retry_success(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM("bad", '{"type":"final","answer":"纠正成功"}')
            result = self.make_runtime(llm, d).run("x", session_id="protocol")
            self.assertEqual(result.answer, "纠正成功")
            self.assertTrue(any(e["event"] == "protocol_retry" for e in result.events))

    def test_persistence_across_runtime_instances(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "db.sqlite")
            first = AgentRuntime(
                ScriptedLLM('{"type":"final","answer":"第一轮"}'),
                store=SessionStore(path),
                trace_dir=str(Path(d) / "traces"),
            )
            first.run("第一轮上下文", session_id="persist")
            second_llm = ScriptedLLM('{"type":"final","answer":"看到了"}')
            AgentRuntime(
                second_llm,
                store=SessionStore(path),
                trace_dir=str(Path(d) / "traces"),
            ).run("继续", session_id="persist")
            self.assertTrue(any(m["content"] == "第一轮上下文" for m in second_llm.messages[0]))

    def test_user_id_ownership(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = self.make_runtime(ScriptedLLM('{"type":"final","answer":"ok"}'), d)
            runtime.run("x", session_id="owned", user_id="alice")
            with self.assertRaises(PermissionError):
                runtime.run("y", session_id="owned", user_id="bob")

    def test_explicit_memory_is_saved_and_recalled_by_user(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "db.sqlite")
            memory = MemoryStore(path)
            AgentRuntime(
                ScriptedLLM('{"type":"final","answer":"已记住"}'),
                store=SessionStore(path),
                memory_store=memory,
                trace_dir=str(Path(d) / "traces"),
            ).run("记住我优先远程岗位", session_id="m1", user_id="alice")
            result = AgentRuntime(
                ScriptedLLM('{"type":"final","answer":"知道"}'),
                store=SessionStore(path),
                memory_store=memory,
                trace_dir=str(Path(d) / "traces"),
            ).run("远程岗位", session_id="m2", user_id="alice")
            self.assertEqual(result.memories_recalled, ["优先远程岗位"])

    def test_memory_is_not_shared_between_users(self):
        with tempfile.TemporaryDirectory() as d:
            memory = MemoryStore(str(Path(d) / "db.sqlite"))
            memory.remember("alice", "喜欢远程工作")
            self.assertEqual(memory.recall("bob", "远程工作"), [])

    def test_missing_user_id_does_not_write_memory(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "db.sqlite")
            memory = MemoryStore(path)
            AgentRuntime(
                ScriptedLLM('{"type":"final","answer":"ok"}'),
                store=SessionStore(path),
                memory_store=memory,
                trace_dir=str(Path(d) / "traces"),
            ).run("记住我喜欢咖啡", session_id="m3")
            self.assertEqual(memory.list(""), [])

    def test_confirmation_blocks_expense_write(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"add","entry_type":"expense","amount":20,"category":"餐饮"}}}'
            )
            result = self.make_runtime(llm, d, confirm_writes=True).run("记录餐饮支出", session_id="confirm")
            self.assertEqual(result.status, "confirmation_required")
            self.assertEqual(
                SessionStore(str(Path(d) / "db.sqlite")).load("confirm")["state"].get("expenses", []),
                [],
            )

    def test_confirmation_does_not_block_expense_list(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"list"}}}',
                '{"type":"final","answer":"空账单"}',
            )
            result = self.make_runtime(llm, d, confirm_writes=True).run("查看支出", session_id="list-confirm")
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.tools_called, ["expense_tracker"])

    def test_confirmed_expense_write_is_executed(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"add","entry_type":"expense","amount":20,"category":"餐饮","confirmed":true}}}',
                '{"type":"final","answer":"完成"}',
            )
            result = self.make_runtime(llm, d, confirm_writes=True).run("确认记录", session_id="confirmed")
            self.assertEqual(result.tools_called, ["expense_tracker"])

    def test_write_operation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            args = '{"action":"add","entry_type":"expense","amount":20,"category":"餐饮","operation_id":"op-1"}'
            llm = ScriptedLLM(
                f'{{"type":"tool_call","tool_call":{{"name":"expense_tracker","arguments":{args}}}}}',
                f'{{"type":"tool_call","tool_call":{{"name":"expense_tracker","arguments":{args}}}}}',
                '{"type":"final","answer":"完成"}',
            )
            result = self.make_runtime(llm, d).run("重复记账", session_id="idem")
            state = SessionStore(str(Path(d) / "db.sqlite")).load("idem")["state"]
            self.assertEqual(len(state["expenses"]), 1)
            self.assertTrue(any(e["event"] == "tool_idempotent_replay" for e in result.events))

    def test_context_compaction(self):
        session = {"messages": [{"role": "user", "content": "x" * 100} for _ in range(8)], "summary": "", "state": {}}
        self.assertTrue(ContextManager(300, 2).compact(session))
        self.assertEqual(len(session["messages"]), 2)

    def test_compaction_preserves_state(self):
        session = {
            "messages": [{"role": "user", "content": "x" * 100} for _ in range(8)],
            "summary": "",
            "state": {"expenses": [{"amount": 1}]},
        }
        ContextManager(300, 2).compact(session)
        self.assertEqual(session["state"]["expenses"][0]["amount"], 1)

    def test_no_compaction_under_limit(self):
        session = {"messages": [{"role": "user", "content": "short"}], "summary": "", "state": {}}
        self.assertFalse(ContextManager(1000).compact(session))

    def test_max_loop_limit(self):
        with tempfile.TemporaryDirectory() as d:
            repeated = '{"type":"tool_call","tool_call":{"name":"weather","arguments":{"city":"上海"}}}'
            result = self.make_runtime(
                ScriptedLLM(repeated, repeated, repeated),
                d,
                max_loops=2,
            ).run("天气", session_id="loops")
            self.assertEqual(result.status, "max_loops")

    def test_completion_guard_stops_repeated_final_without_progress(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.make_runtime(
                ScriptedLLM(
                    '{"type":"final","answer":"上海天气需要查询"}',
                    '{"type":"final","answer":"仍然需要查询"}',
                ),
                d,
                max_loops=3,
            ).run("查询上海天气", session_id="stalled")
            self.assertEqual(result.status, "stalled")
            self.assertEqual(result.loops, 2)
            self.assertTrue(any(event["event"] == "completion_stalled" for event in result.events))

    def test_income_and_expense_batch_then_today_net_change(self):
        with tempfile.TemporaryDirectory() as d:
            today = date.today().isoformat()
            llm = ScriptedLLM(
                json.dumps(
                    {
                        "type": "tool_calls",
                        "tool_calls": [
                            {
                                "name": "expense_tracker",
                                "arguments": {
                                    "action": "add",
                                    "entry_type": "expense",
                                    "amount": 23,
                                    "category": "交通",
                                    "expense_date": today,
                                },
                            },
                            {
                                "name": "expense_tracker",
                                "arguments": {
                                    "action": "add",
                                    "entry_type": "income",
                                    "amount": 45,
                                    "category": "收入",
                                    "expense_date": today,
                                },
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                '{"type":"final","answer":"已记录今天的收入和支出"}',
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool_call": {"name": "expense_tracker", "arguments": {"action": "summary", "expense_date": today}},
                    },
                    ensure_ascii=False,
                ),
                '{"type":"final","answer":"今天净变化为 +22 元"}',
            )
            runtime = self.make_runtime(llm, d)
            runtime.run("记录今天交通支出23，收入45", session_id="daily-balance")
            result = runtime.run("今天余额变化", session_id="daily-balance")
            summary = json.loads(llm.messages[-1][-1]["content"])
            state = SessionStore(str(Path(d) / "db.sqlite")).load("daily-balance")["state"]
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.tools_called, ["expense_tracker"])
            self.assertEqual(len(state["expenses"]), 2)
            self.assertEqual(summary["income_total"], 45.0)
            self.assertEqual(summary["expense_total"], 23.0)
            self.assertEqual(summary["net_change"], 22.0)
            self.assertIn(today, llm.messages[0][0]["content"])

    def test_completion_guard_requires_two_entries_for_income_and_expense(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"add","entry_type":"expense","amount":23,"category":"交通"}}}',
                '{"type":"final","answer":"已记录"}',
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"add","entry_type":"income","amount":45,"category":"收入"}}}',
                '{"type":"final","answer":"已完整记录收支"}',
            )
            result = self.make_runtime(llm, d, max_loops=5).run("记录今天交通支出23，收入45", session_id="two-entries")
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.tools_called, ["expense_tracker", "expense_tracker"])
            self.assertTrue(any(event["event"] == "completion_guard" for event in result.events))

    def test_trace_file_contains_lifecycle_events(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.make_runtime(
                ScriptedLLM('{"type":"final","reasoning_summary":"直接回答","answer":"ok"}'),
                d,
            ).run("x", session_id="trace")
            trace_path = Path(d) / "traces" / f"{result.trace_id}.jsonl"
            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            names = {item["event"] for item in events}
            self.assertIn("request_started", names)
            self.assertIn("final_answer", names)

    def test_llm_timeout_is_retried(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.make_runtime(
                FlakyLLM('{"type":"final","answer":"重试后成功"}'),
                d,
                llm_retries=1,
                retry_backoff=0,
            ).run("重试", session_id="retry")
            self.assertEqual(result.llm_calls, 2)
            self.assertEqual(result.answer, "重试后成功")

    def test_result_contains_observability_fields(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.make_runtime(ScriptedLLM('{"type":"final","answer":"ok"}'), d).run("x", session_id="metrics")
            self.assertGreaterEqual(result.llm_calls, 1)
            self.assertGreaterEqual(result.total_latency_ms, 0)
            self.assertTrue(any(e["event"] == "final_answer" and "latency_ms" in e for e in result.events))

    def test_empty_input_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                self.make_runtime(ScriptedLLM('{"type":"final","answer":"ok"}'), d).run(" ", session_id="empty")

    def test_expense_missing_required_fields_is_handled(self):
        with tempfile.TemporaryDirectory() as d:
            llm = ScriptedLLM(
                '{"type":"tool_call","tool_call":{"name":"expense_tracker","arguments":{"action":"add","entry_type":"expense","amount":5}}}',
                '{"type":"final","answer":"缺少分类"}',
            )
            result = self.make_runtime(llm, d).run("记账", session_id="missing-fields")
            self.assertEqual(result.status, "completed")
            self.assertTrue(any(e["event"] == "tool_failed" for e in result.events))

    def test_calculator_rejects_large_power(self):
        with self.assertRaises(ToolExecutionError):
            calculator_tool().invoke({"expression": "2 ** 99"}, session_state={})


if __name__ == "__main__":
    unittest.main()
