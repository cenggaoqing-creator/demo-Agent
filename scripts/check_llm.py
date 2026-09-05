"""Check OpenAI-compatible endpoint connectivity without printing the API key."""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_env() -> None:
    path = Path.cwd() / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    load_env()
    key = os.getenv("AGENT_LLM_API_KEY", "")
    base_url = os.getenv("AGENT_LLM_BASE_URL", "").rstrip("/")
    model = os.getenv("AGENT_LLM_MODEL", "")
    if not key or not base_url or not model:
        print("配置不完整：需要 AGENT_LLM_API_KEY、AGENT_LLM_BASE_URL、AGENT_LLM_MODEL")
        return 2
    url = base_url + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read(500).decode("utf-8", errors="replace")
            print(f"连接成功: HTTP {response.status}, model={model}")
            print(body)
            return 0
    except urllib.error.HTTPError as exc:
        body = exc.read(500).decode("utf-8", errors="replace")
        print(f"服务已连接但返回 HTTP {exc.code}: {body}")
        return 1
    except Exception as exc:
        print(f"连接失败: {exc.__class__.__name__}: {exc}")
        print(f"检查地址、网络代理和防火墙: {url}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

