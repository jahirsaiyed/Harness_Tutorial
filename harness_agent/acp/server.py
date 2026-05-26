"""Minimal ACP-style stdio JSON-RPC server for Harness Agent."""

from __future__ import annotations

import json
import sys
from typing import Any

from harness_agent.agent import AIAgent


def run_acp_stdio() -> None:
    """Read JSON lines from stdin, write JSON lines to stdout."""
    agent = AIAgent()
    session_id: str | None = None
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method")
        req_id = req.get("id")
        if method == "initialize":
            _reply(req_id, {"capabilities": {"chat": True}})
            continue
        if method == "chat/send":
            params = req.get("params", {})
            text = params.get("message", "")
            result = agent.run_conversation(text, session_id=session_id)
            session_id = result.session_id
            _reply(req_id, {"message": result.assistant_text, "session_id": session_id})
            continue
        _reply(req_id, {"error": f"unknown method: {method}"})


def _reply(req_id: Any, result: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps({"id": req_id, "result": result}) + "\n")
    sys.stdout.flush()
