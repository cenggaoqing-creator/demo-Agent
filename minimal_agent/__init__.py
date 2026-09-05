from pathlib import Path
import os


def _load_local_env() -> None:
    """Load a tiny .env subset without adding a dotenv dependency."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env()

from .runtime import AgentRuntime, AgentResult
from .llm import OpenAICompatibleLLM
from .tools.base import ToolRegistry

__all__ = ["AgentRuntime", "AgentResult", "OpenAICompatibleLLM", "ToolRegistry"]
