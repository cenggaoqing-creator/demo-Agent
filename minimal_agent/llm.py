from __future__ import annotations

import os
from typing import Any


class OpenAICompatibleLLM:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None, timeout: float | None = None) -> None:
        self.api_key = api_key or os.getenv("AGENT_LLM_API_KEY", "")
        self.base_url = base_url or os.getenv("AGENT_LLM_BASE_URL", "")
        self.model = model or os.getenv("AGENT_LLM_MODEL", "")
        self.timeout = float(os.getenv("AGENT_LLM_TIMEOUT", "20")) if timeout is None else timeout
        self.max_output_tokens = int(os.getenv("AGENT_MAX_OUTPUT_TOKENS", "800"))
        self.disable_thinking = os.getenv("AGENT_DISABLE_THINKING", "false").lower() == "true"

    def complete(self, messages: list[dict[str, Any]]) -> str:
        if not self.api_key or not self.model:
            raise RuntimeError("未配置 AGENT_LLM_API_KEY 或 AGENT_LLM_MODEL")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 依赖，请 pip install -r requirements.txt") from exc
        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout)
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        # SiliconFlow exposes Qwen thinking control through extra_body.
        if self.disable_thinking:
            request_kwargs["extra_body"] = {"enable_thinking": False}
        response = client.chat.completions.create(**request_kwargs)
        return response.choices[0].message.content or ""
