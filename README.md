# 从零实现的最小 Agent

## 运行方式

在 VS Code 的 CMD 终端中打开项目根目录，然后执行：

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
```

在 `.env` 中填写真实的 OpenAI-compatible 模型配置，且不要提交密钥：

```env
AGENT_LLM_API_KEY=
AGENT_LLM_BASE_URL=
AGENT_LLM_MODEL=
```

运行测试并启动服务：

```cmd
python -m unittest discover -s tests -v
python -m uvicorn minimal_agent.api:app --host 127.0.0.1 --port 8010 --reload
```

## 系统设计

核心 Runtime 不依赖 LangGraph、LangChain、OpenHands 或 OpenClaw。一次请求按以下状态流转执行：

1. 使用 `session_id` 加载会话状态，并校验其归属的 `user_id`。
2. 由 Context Builder 组合当前输入、会话历史、历史摘要、长期 Memory 和工具 Schema，调用 LLM。
3. 解析 LLM 返回的 JSON 决策：直接回答，或请求一个/多个工具调用。
4. 通过工具注册器查找 `ToolSpec`，使用 Pydantic 参数模型校验 arguments，执行对应 handler。
5. 将工具调用和结果写回 session，继续下一轮；收到 `final` 后经 Completion Guard 检查，再返回最终答案。

工具注册项包含名称、描述、参数 Schema、读写副作用标记和确定性 handler。SQLite 持久化 session 的消息、摘要与工具 state；不同 `session_id` 代表不同窗口，保证窗口间上下文和账本状态隔离。每个请求的执行过程写入 JSONL trace，便于定位模型、协议和工具异常。

## Memory 的召回时机与放置方式

短期记忆放在 session 中：当前输入、最近用户/助手消息、上轮工具结果和历史摘要构成模型 Context。每次调用 LLM 前构造该 Context，因此纯对话追问可以使用前文，带工具的追问可以使用已经执行过的工具结果。

长期 Memory 只在用户明确提出“记住……”等持久偏好时写入，并以 `user_id` 隔离；每次构建 Context 时按当前用户召回。消息过长时保留最近消息，将早期消息压缩成摘要。收支账本等结构化业务事实保存在 session 的工具 state 中，仅在 handler 执行时从 SQLite 读取，不依赖模型在自然语言上下文中记住。

## AI Prompt 与问题解决记录

AI 主要用于需求拆解、反例生成、协议审查、测试草拟和文档润色；核心 Runtime 状态边界、工具副作用策略、Session/Memory 分层和最终代码审查由我自己确定。使用 AI 时，我要求它先给出可验证的方案和风险清单，再生成局部实现，避免把整个项目当成一次性代码生成任务。

关键 Prompt 包括：

```text
请把题目拆成 Runtime 状态机、LLM 输出协议、工具注册、Session/Memory/Context 和测试验收项，并为每项给出可执行的行为契约。

请从协议歧义、重复副作用、session 污染、无限循环和敏感信息泄露角度生成反例。每个反例给出触发输入、预期行为和测试断言，不要直接修改代码。

请审查这个 Runtime：模型返回 final 但没有执行工具时会发生什么？复合请求如何表达多个工具调用？请指出无进展循环和重复写入风险。
```

### 设计取舍
- **使用 SQLite 保存状态**：SQLite 是 Python 标准库能力，足以覆盖 session 持久化、窗口隔离和服务重启恢复，不需要额外部署 PostgreSQL 或 Redis。
- **不保存原始思考过程**：trace 只记录简短的 `reasoning_summary`。后续行为依赖用户消息、工具结果和结构化工具 state，而不是把隐藏思考过程写入长期 Memory 或返回给用户。
- **工具保持离线且可测试**：`calculator` 使用 AST 白名单而不是 `eval`；`search` 查询固定 mock 语料并限制 `top_k`；`weather` 模拟外部查询；`expense_tracker` 展示有状态读写。这样每个工具的边界和副作用都能通过确定性测试验证。

### 真实故障复盘：余额追问跑满最大轮次

**复现步骤**：在同一个 `session_id` 依次发送“记录今天交通支出 23，收入 45”和“今天余额变化”。预期结果是两笔账都写入，第二次返回当天净变化 `+22`。

**实际现象**：请求没有因为 LLM API 超时失败，而是 Runtime 连续多轮收到 `final`。Completion Guard 把“今天余额变化”错误判成了无关工具，Runtime 每轮都追加相同提醒并继续，直到达到 `AGENT_MAX_LOOPS=8`。trace 因此表现为“模型有响应、状态却没有进展”的循环。

排查同一条链路时又发现两个会造成错误数据的独立缺陷：

1. 原协议只有单个 `tool_call` 字段。模型用重复 JSON 键表达收入和支出时，标准解析器只保留最后一个键，交通支出会被静默丢弃。
2. 账本只把金额当作正数累加，即使两笔都写成功，也会算出 `23 + 45 = 68`，而不是收入减支出的净变化 `45 - 23 = +22`。

**修复与验证**：

| 层面 | 修复 | 回归断言 |
| --- | --- | --- |
| 完成守卫 | 仅将明确数学表达式映射到 `calculator`，检索映射到 `search`；收入、支出和余额映射到 `expense_tracker`，并识别“同时记录收入和支出”为两次写入 | “今天余额变化”会请求账本汇总，不再被判为无关工具 |
| 输出协议 | 增加有序 `tool_calls` 数组；解析阶段拒绝重复 JSON 键，同时兼容旧的单调用格式 | 一次请求中的收入、支出按数组顺序各执行一次 |
| 账本语义 | 增加 `entry_type`，分别汇总 `income_total`、`expense_total` 和 `net_change`；旧记录无类型时按支出解释 | 23 支出、45 收入的净变化为 `+22`，历史 session 仍可读取 |
| 日期与幂等 | Context 注入当前 ISO 日期；写操作使用 `operation_id` 防止重试重复记账 | “今天”查询生成合法日期；相同操作重试不会新增记录 |
| 成本控制 | 相同缺少工具的 `final` 连续两轮后熔断，返回 `stalled` 并记录 `completion_stalled` | 无进展请求不会消耗完 8 轮上限 |

这次修复覆盖协议歧义、业务语义、编排守卫和成本控制四个层面，并补充了对应回归测试；目标不是给某一句输入加特例，而是让同类请求在协议、状态和重试场景下都保持一致。

