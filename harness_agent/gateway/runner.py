"""Gateway runner — HTTP webhook + test adapter."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs

from harness_agent.agent import AIAgent


class GatewayRunner:
    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self.host = host or os.environ.get("HARNESS_GATEWAY_HOST", "127.0.0.1")
        self.port = port or int(os.environ.get("HARNESS_GATEWAY_PORT", "8765"))
        self.agent = AIAgent()
        self._sessions: dict[str, str] = {}

    def handle_message(self, platform: str, user_id: str, text: str) -> str:
        key = f"{platform}:{user_id}"
        session_id = self._sessions.get(key)
        result = self.agent.run_conversation(text, session_id=session_id)
        if result.session_id:
            self._sessions[key] = result.session_id
        return result.assistant_text

    def run_http(self) -> None:
        runner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    data = {"text": body}
                text = data.get("text", "")
                user = data.get("user_id", "default")
                platform = data.get("platform", "webhook")
                reply = runner.handle_message(platform, user, text)
                payload = json.dumps({"reply": reply}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        server = HTTPServer((self.host, self.port), Handler)
        print(f"Harness Agent gateway on http://{self.host}:{self.port}")
        server.serve_forever()
