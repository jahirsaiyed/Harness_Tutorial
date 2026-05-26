"""Export sessions to ShareGPT-style JSONL trajectories."""

from __future__ import annotations

import json
from pathlib import Path

from harness_agent.config import get_config
from harness_agent.sessions.store import SessionStore


def export_trajectories(output: Path) -> int:
    store = SessionStore()
    count = 0
    with store._connect() as conn:  # noqa: SLF001
        sessions = conn.execute("SELECT id FROM sessions").fetchall()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in sessions:
            sid = row["id"]
            messages = store.load_messages(sid)
            convo = []
            for m in messages:
                if m.role in ("user", "assistant") and m.content:
                    convo.append({"from": "human" if m.role == "user" else "gpt", "value": m.content})
            if convo:
                fh.write(json.dumps({"conversations": convo, "session_id": sid}) + "\n")
                count += 1
    return count
