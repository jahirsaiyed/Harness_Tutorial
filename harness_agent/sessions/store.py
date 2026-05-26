"""SQLite session storage with FTS5 search."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from harness_agent.config import get_config
from harness_agent.observations import wrap_result
from harness_agent.tools.registry import register
from harness_agent.types import Message


@dataclass
class SessionMeta:
    id: str
    title: str
    parent_id: str | None = None


class SessionStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or get_config().db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    parent_id TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    payload TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                    session_id, content, content='turns', content_rowid='id'
                );
                """
            )
            conn.commit()

    def create_session(self, title: str = "New session", parent_id: str | None = None) -> str:
        sid = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, parent_id) VALUES (?, ?, ?)",
                (sid, title, parent_id),
            )
            conn.commit()
        return sid

    def append_turn(self, session_id: str, message: Message) -> None:
        payload = json.dumps(message.to_openai())
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO turns (session_id, role, content, payload) VALUES (?, ?, ?, ?)",
                (session_id, message.role, message.content or "", payload),
            )
            rowid = cur.lastrowid
            conn.execute(
                "INSERT INTO turns_fts (rowid, session_id, content) VALUES (?, ?, ?)",
                (rowid, session_id, message.content or ""),
            )
            conn.commit()

    def load_messages(self, session_id: str) -> list[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        messages: list[Message] = []
        for row in rows:
            data = json.loads(row["payload"])
            messages.append(
                Message(
                    role=data["role"],
                    content=data.get("content"),
                    tool_calls=data.get("tool_calls"),
                    tool_call_id=data.get("tool_call_id"),
                    name=data.get("name"),
                )
            )
        return messages

    def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, content FROM turns_fts
                WHERE turns_fts MATCH ?
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [{"session_id": r["session_id"], "snippet": r["content"]} for r in rows]


def search_sessions(query: str) -> str:
    store = SessionStore()
    hits = store.search(query)
    return wrap_result(
        status="success",
        summary=f"Found {len(hits)} matches",
        artifacts=[h["session_id"] for h in hits],
        detail=json.dumps(hits, indent=2),
    )


register(
    name="search_sessions",
    description="Full-text search prior conversation turns.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    handler=search_sessions,
    toolset="sessions",
)
