"""Sandboxed shell tool."""

from __future__ import annotations

from harness_agent.observations import wrap_result
from harness_agent.tools.environments import get_terminal_backend
from harness_agent.tools.registry import register


def run_shell(command: str) -> str:
    backend = get_terminal_backend()
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
