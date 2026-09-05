from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self, path: str = "runtime/minimal_agent.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._db() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, user_id TEXT, messages TEXT NOT NULL, summary TEXT NOT NULL, state TEXT NOT NULL, turn_count INTEGER NOT NULL, updated_at TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _db(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def load(self, session_id: str, user_id: str | None = None) -> dict[str, Any]:
        with self._lock, self._db() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                data = {"session_id": session_id, "user_id": user_id, "messages": [], "summary": "", "state": {}, "turn_count": 0}
                self.save(data)
                return data
            if user_id and row["user_id"] and user_id != row["user_id"]:
                raise PermissionError("session 不属于当前 user_id")
            return {"session_id": row["session_id"], "user_id": row["user_id"], "messages": json.loads(row["messages"]), "summary": row["summary"], "state": json.loads(row["state"]), "turn_count": row["turn_count"]}

    def save(self, data: dict[str, Any]) -> None:
        with self._lock, self._db() as conn:
            conn.execute("INSERT INTO sessions(session_id,user_id,messages,summary,state,turn_count,updated_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET user_id=excluded.user_id,messages=excluded.messages,summary=excluded.summary,state=excluded.state,turn_count=excluded.turn_count,updated_at=excluded.updated_at", (data["session_id"], data.get("user_id"), json.dumps(data.get("messages", []), ensure_ascii=False), data.get("summary", ""), json.dumps(data.get("state", {}), ensure_ascii=False), data.get("turn_count", 0), _now()))

    def delete(self, session_id: str) -> None:
        with self._lock, self._db() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
