from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .llm import OpenAICompatibleLLM
from .memory import MemoryStore
from .runtime import AgentRuntime
from .session import SessionStore


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    user_id: str | None = None


store = SessionStore(os.getenv("AGENT_DB_PATH", "runtime/minimal_agent.db"))
memory_store = MemoryStore(os.getenv("AGENT_DB_PATH", "runtime/minimal_agent.db"))
runtime = AgentRuntime(OpenAICompatibleLLM(), store=store, memory_store=memory_store)
app = FastAPI(title="最小 Agent Runtime", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "api_key_configured": bool(os.getenv("AGENT_LLM_API_KEY")), "model_configured": bool(os.getenv("AGENT_LLM_MODEL")), "confirm_writes": runtime.confirm_writes, "max_loops": runtime.max_loops}


@app.post("/v1/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        result = runtime.run(request.message, session_id=request.session_id, user_id=request.user_id)
        if result.status == "failed":
            raise HTTPException(status_code=503, detail={"message": result.answer, "trace_id": result.trace_id})
        return {"answer": result.answer, "session_id": result.session_id, "trace_id": result.trace_id, "loops": result.loops, "tools_called": result.tools_called, "status": result.status, "llm_calls": result.llm_calls, "total_latency_ms": result.total_latency_ms, "memories_recalled": result.memories_recalled}
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/sessions/{session_id}")
def session(session_id: str) -> dict[str, Any]:
    return store.load(session_id)


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    store.delete(session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/v1/memories/{user_id}")
def memories(user_id: str) -> dict[str, Any]:
    return {"user_id": user_id, "memories": memory_store.list(user_id)}


@app.delete("/v1/memories/{user_id}/{memory_id}")
def delete_memory(user_id: str, memory_id: int) -> dict[str, Any]:
    return {"user_id": user_id, "memory_id": memory_id, "deleted": memory_store.delete(user_id, memory_id)}


@app.get("/v1/traces/{trace_id}")
def trace(trace_id: str) -> dict[str, Any]:
    """读取一次请求的结构化执行轨迹。"""
    if not trace_id.isalnum():
        raise HTTPException(status_code=400, detail="非法 trace_id")
    path = Path(os.getenv("AGENT_TRACE_DIR", "runtime/traces")) / f"{trace_id}.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="trace 不存在")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"trace_id": trace_id, "events": events}
