from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class Trace:
    def __init__(self, trace_dir: str, session_id: str) -> None:
        self.trace_id = uuid.uuid4().hex
        self.session_id = session_id
        self.events: list[dict[str, Any]] = []
        self.path = Path(trace_dir) / f"{self.trace_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **payload: Any) -> None:
        item = {"ts": time.time(), "trace_id": self.trace_id, "session_id": self.session_id, "event": event, **payload}
        self.events.append(item)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

