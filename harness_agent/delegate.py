"""Subagent delegation tool."""

from __future__ import annotations

import json

from harness_agent.observations import wrap_result
from harness_agent.tools.registry import register


def delegate_subagent(task: str) -> str:
    from harness_agent.agent import AIAgent

    child = AIAgent(isolated=True)
    result = child.run_conversation(task, session_id=None)
    return wrap_result(
        status="success",
        summary="Subagent completed",
        detail=result.assistant_text,
        artifacts=[result.session_id] if result.session_id else [],
    )


register(
    name="delegate_subagent",
    description="Spawn an isolated subagent with a fresh context to handle a subtask.",
    parameters={
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "required": ["task"],
    },
    handler=delegate_subagent,
    toolset="delegate",
)
