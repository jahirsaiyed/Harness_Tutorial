"""LangGraph state definitions for Harness Agent.

Two state types are defined:

  AgentState       — single-agent graph (replaces AIAgent's for-loop)
  SupervisorState  — multi-agent supervisor + worker graph (DeepAgents pattern)

Both use a list-append reducer for messages so the graph accumulates the full
conversation history without overwriting on each node transition.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from harness_agent.types import Message


def _append_messages(left: list[Message], right: list[Message]) -> list[Message]:
    """Reducer: append incoming messages to the existing list."""
    return left + right


class AgentState(TypedDict):
    """State flowing through the single-agent graph nodes."""

    # Full conversation history — each node appends, never overwrites
    messages: Annotated[list[Message], _append_messages]
    session_id: str | None
    tool_call_count: int
    had_error: bool
    # Set by call_model when it produces a final (non-tool) answer
    final_text: str


class SupervisorState(TypedDict):
    """State for the multi-agent supervisor graph.

    The supervisor sets next_worker on each turn; workers append to
    worker_results and then return control to the supervisor.
    """

    messages: Annotated[list[Message], _append_messages]
    session_id: str | None
    # The high-level task given to the orchestrator
    task: str
    # Supervisor sets this to route to the next worker (or 'FINISH')
    next_worker: str
    # Each worker appends a one-line summary of what it did/found
    worker_results: Annotated[list[str], operator.add]
    final_text: str
    # Safety counter so the supervisor loop terminates
    turn: int
