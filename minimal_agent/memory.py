"""Explicit, user-scoped long-term memory.

Only facts the user explicitly asks us to remember are written here. Session
messages remain window-scoped and are stored by SessionStore.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(self, path: str = "runtime/minimal_agent.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._db() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memories ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, "
                "content TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                "UNIQUE(user_id, content))"
            )

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def remember(self, user_id: str, content: str) -> dict[str, Any]:
        content = content.strip()
        if not user_id or not content:
            raise ValueError("user_id 和 memory content 不能为空")
        with self._lock, self._db() as conn:
            conn.execute(
                "INSERT INTO memories(user_id, content, created_at, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(user_id, content) DO UPDATE SET updated_at=excluded.updated_at",
                (user_id, content, _now(), _now()),
            )
            row = conn.execute("SELECT * FROM memories WHERE user_id=? AND content=?", (user_id, content)).fetchone()
            return dict(row)

    def list(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._db() as conn:
            rows = conn.execute("SELECT * FROM memories WHERE user_id=? ORDER BY updated_at DESC LIMIT ?", (user_id, limit)).fetchall()
            return [dict(row) for row in rows]

    def recall(self, user_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not user_id or not query:
            return []
        normalized = query.lower()
        terms = {normalized[i : i + 2] for i in range(max(0, len(normalized) - 1)) if normalized[i : i + 2].strip()}
        candidates = self.list(user_id, limit=100)
        scored = []
        for item in candidates:
            content = item["content"].lower()
            score = int(normalized in content) * 10 + sum(term in content for term in terms)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def delete(self, user_id: str, memory_id: int) -> bool:
        with self._lock, self._db() as conn:
            cur = conn.execute("DELETE FROM memories WHERE user_id=? AND id=?", (user_id, memory_id))
            return cur.rowcount > 0

