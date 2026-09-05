from __future__ import annotations

import argparse

from .llm import OpenAICompatibleLLM
from .runtime import AgentRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="最小 Agent CLI")
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()
    runtime = AgentRuntime(OpenAICompatibleLLM())
    session_id = args.session_id
    print("输入 exit 退出。")
    while True:
        try:
            message = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"exit", "quit"}:
            break
        result = runtime.run(message, session_id=session_id)
        session_id = result.session_id
        print(f"Agent> {result.answer}\n[session={session_id} trace={result.trace_id}]" )


if __name__ == "__main__":
    main()

