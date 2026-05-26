"""LangGraph node functions for Harness Agent.

Nodes are plain functions (state: S) -> dict that return partial state updates.
LangGraph merges the returned dict into the current state using the registered
reducers — messages use list-append, scalars overwrite.

Two node families:

  call_model / execute_tools   — single-agent loop nodes
  supervisor / worker          — multi-agent supervisor-pattern nodes
"""

from __future__ import annotations

import json
from typing import Any

from harness_agent.compression.summarize import compress_messages
from harness_agent.graph.state import AgentState, SupervisorState
from harness_agent.providers.registry import get_provider_registry
from harness_agent.tools.registry import get_registry
from harness_agent.types import Message

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tool_calls_to_openai(calls: list) -> list[dict[str, Any]]:
    """Convert internal ToolCall dataclasses to OpenAI wire-format dicts."""
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments),
            },
        }
        for tc in calls
    ]


# ---------------------------------------------------------------------------
# Single-agent nodes
# ---------------------------------------------------------------------------

def make_call_model_node(
    *,
    provider: str | None = None,
    model: str | None = None,
    toolsets: list[str] | None = None,
) -> Any:
    """Return a call_model node bound to a specific provider/model/toolsets.

    On each invocation:
      1. Compresses messages if over token budget.
      2. Calls the LLM with available tool schemas.
      3. If the model returned tool_calls → appends the assistant message,
         returns it so execute_tools can run next.
      4. If the model returned plain text → sets final_text and terminates.
    """
    prov, resolved_model = get_provider_registry().resolve(provider, model)
    registry = get_registry()

    def call_model(state: AgentState) -> dict:
        messages = compress_messages(list(state["messages"]))
        schemas = registry.openai_schemas(toolsets)
        text, calls = prov.complete_with_tools(messages, schemas, model=resolved_model)

        if calls:
            asst = Message(
                role="assistant",
                content=text,
                tool_calls=_tool_calls_to_openai(calls),
            )
            return {"messages": [asst]}

        final = text or ""
        return {
            "messages": [Message(role="assistant", content=final)],
            "final_text": final,
        }

    return call_model


def make_execute_tools_node(toolsets: list[str] | None = None) -> Any:
    """Return an execute_tools node that dispatches all pending tool calls.

    Reads the last assistant message, runs each requested tool via the
    registry, and appends tool-role messages with the observations.
    """
    registry = get_registry()

    def execute_tools(state: AgentState) -> dict:
        last = state["messages"][-1]
        if not last.tool_calls:
            return {}

        tool_msgs: list[Message] = []
        had_error = state.get("had_error", False)
        count = state.get("tool_call_count", 0)

        for tc_dict in last.tool_calls:
            fn_name = tc_dict["function"]["name"]
            raw_args = tc_dict["function"]["arguments"]
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

            result = registry.dispatch(fn_name, args)
            if '"status": "error"' in result:
                had_error = True
            count += 1

            tool_msgs.append(
                Message(
                    role="tool",
                    content=result,
                    tool_call_id=tc_dict["id"],
                    name=fn_name,
                )
            )

        return {
            "messages": tool_msgs,
            "tool_call_count": count,
            "had_error": had_error,
        }

    return execute_tools


def should_continue(state: AgentState) -> str:
    """Conditional edge: route to execute_tools or END.

    The last message in state determines the route:
      - assistant message with tool_calls → "execute_tools"
      - anything else (final text)         → "END"
    """
    last = state["messages"][-1]
    if last.role == "assistant" and last.tool_calls:
        return "execute_tools"
    return "END"


# ---------------------------------------------------------------------------
# Multi-agent supervisor nodes
# ---------------------------------------------------------------------------

# Workers the supervisor may delegate to, plus the FINISH sentinel
WORKERS = ["researcher", "coder", "planner"]

WORKER_PERSONAS = {
    "researcher": (
        "You are a research specialist. Use file-reading and search tools to "
        "gather information and return a concise summary of what you found."
    ),
    "coder": (
        "You are a coding specialist. Write code, execute it in the sandbox, "
        "and report the result. Prefer short, focused scripts."
    ),
    "planner": (
        "You are a planning specialist. Break down complex goals into numbered "
        "steps. Do not call tools — reason only."
    ),
}

WORKER_TOOLSETS: dict[str, list[str]] = {
    "researcher": ["files", "sessions"],
    "coder": ["terminal", "files"],
    "planner": [],
}

SUPERVISOR_SYSTEM = (
    "You are a supervisor orchestrating a team of specialist agents.\n"
    "Decide which worker should act next based on the task and progress so far.\n\n"
    "Workers:\n"
    "  researcher — reads files, searches, gathers information\n"
    "  coder      — writes and runs code in a sandboxed environment\n"
    "  planner    — breaks down tasks and reasons without tools\n\n"
    "Respond with ONLY the worker name, or FINISH when the task is complete.\n"
    "Do not explain your choice."
)


def make_supervisor_node(
    *,
    provider: str | None = None,
    model: str | None = None,
) -> Any:
    """Return a supervisor node that routes the task to the next worker.

    The supervisor sees: the original task + accumulated worker_results.
    It responds with a single word: a worker name or 'FINISH'.
    """
    prov, resolved_model = get_provider_registry().resolve(provider, model)

    def supervisor(state: SupervisorState) -> dict:
        results_so_far = "\n".join(state.get("worker_results") or ["(none yet)"])
        user_content = (
            f"Task: {state['task']}\n\n"
            f"Worker results so far:\n{results_so_far}\n\n"
            "Which worker should act next? (or FINISH)"
        )
        messages = [
            Message(role="system", content=SUPERVISOR_SYSTEM),
            Message(role="user", content=user_content),
        ]
        text, _ = prov.complete_with_tools(messages, [], model=resolved_model)
        choice = (text or "FINISH").strip().split()[0].lower()
        valid = {w.lower() for w in WORKERS} | {"finish"}
        if choice not in valid:
            choice = "FINISH"
        # Normalise case for graph routing
        next_worker = choice.upper() if choice == "finish" else choice
        return {"next_worker": next_worker, "turn": state.get("turn", 0) + 1}

    return supervisor


def make_worker_node(
    worker_name: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> Any:
    """Return a worker node that executes a focused tool loop for one sub-task.

    Each worker:
      1. Receives the shared task description.
      2. Runs its own mini tool loop (max 5 turns) with its allowed toolset.
      3. Appends a one-line summary to worker_results.
      4. Returns control to the supervisor.
    """
    prov, resolved_model = get_provider_registry().resolve(provider, model)
    registry = get_registry()
    persona = WORKER_PERSONAS[worker_name]
    toolsets = WORKER_TOOLSETS[worker_name]

    def worker(state: SupervisorState) -> dict:
        schemas = registry.openai_schemas(toolsets)
        messages: list[Message] = [
            Message(role="system", content=persona),
            Message(role="user", content=state["task"]),
        ]

        final_text = ""
        for _ in range(5):  # per-worker safety limit
            text, calls = prov.complete_with_tools(messages, schemas, model=resolved_model)
            final_text = text or ""
            if not calls:
                break
            asst = Message(
                role="assistant",
                content=text,
                tool_calls=_tool_calls_to_openai(calls),
            )
            messages.append(asst)
            for tc in calls:
                result = registry.dispatch(tc.name, tc.arguments)
                messages.append(
                    Message(role="tool", content=result, tool_call_id=tc.id)
                )

        summary = f"[{worker_name}] {final_text[:300] or '(no output)'}"
        return {
            "worker_results": [summary],
            "messages": messages[2:],  # exclude system + original user msg
        }

    return worker
