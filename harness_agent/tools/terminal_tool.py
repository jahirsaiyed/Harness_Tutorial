"""Sandboxed shell tool.

Routes through the sandbox abstraction layer (harness_agent.sandbox).
The active backend is selected by HARNESS_SANDBOX env var:
  local   — subprocess in workspace (default, no container isolation)
  docker  — ephemeral python:3.11-slim container, network disabled
  e2b     — cloud microVM via e2b.dev (requires E2B_API_KEY)
"""

from __future__ import annotations

from harness_agent.observations import wrap_result
from harness_agent.sandbox.base import get_sandbox
from harness_agent.tools.registry import register


def run_shell(command: str) -> str:
    backend = get_sandbox()
    code, out, err = backend.run(command)
    status = "success" if code == 0 else "error"
    detail = (out + ("\n" + err if err else "")).strip()[:8000]
    return wrap_result(
        status=status,
        summary=f"Exit code {code}",
        next_actions=[] if code == 0 else ["Inspect stderr and adjust command."],
        detail=detail,
    )


register(
    name="run_shell",
    description="Run a shell command in the sandboxed workspace (local or docker backend).",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    handler=run_shell,
    toolset="terminal",
)
