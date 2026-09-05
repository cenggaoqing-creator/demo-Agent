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

详细的面试准备、AI 协作记录和问题解决材料保留在本地 `docs/` 目录。本次 GitHub 发布按要求不包含该目录。
