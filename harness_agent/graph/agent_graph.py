"""LangGraph single-agent graph — replaces AIAgent's manual for-loop.

Graph shape:

    [START]
       │
       ▼
  call_model ──(tool_calls present)──> execute_tools ──┐
       │                                               │
       │ (no tool_calls / final answer)                │
       ▼                                               │
     [END]  <─────────────────────────────────────────┘
            (execute_tools always goes back to call_model)

Benefits over the manual for-loop in agent.py:
  - Observable: every state transition can be streamed or logged
  - Extensible: insert a compression node, a guard node, etc.
  - Testable: nodes are plain functions — easy to unit-test in isolation
  - Supports async: compile(checkpointer=...) for streaming checkpoints
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from harness_agent.graph.nodes import (
    make_call_model_node,
    make_execute_tools_node,
    should_continue,
)
from harness_agent.graph.state import AgentState
from harness_agent.learning.skill_writer import LearningLoop, maybe_write_skill
from harness_agent.prompt.builder import PromptBuilder
from harness_agent.sessions.store import SessionStore
from harness_agent.types import AgentTurnResult, Message


def build_agent_graph(
    *,
    provider: str | None = None,
    model: str | None = None,
    toolsets: list[str] | None = None,
) -> Any:
    """Compile and return a runnable LangGraph agent graph.

    Args:
        provider:  Provider name ('openai', 'anthropic', 'deepseek', 'compass').
                   Falls back to HARNESS_DEFAULT_PROVIDER env var.
        model:     Model identifier. Falls back to HARNESS_DEFAULT_MODEL.
        toolsets:  List of toolset names to expose. None = all registered tools.

    Returns:
        A compiled LangGraph graph that accepts AgentState and returns AgentState.
    """
    call_model = make_call_model_node(provider=provider, model=model, toolsets=toolsets)
    execute_tools = make_execute_tools_node(toolsets=toolsets)

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("execute_tools", execute_tools)

    graph.set_entry_point("call_model")

    # call_model → execute_tools (if tool calls) or END (if final answer)
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {"execute_tools": "execute_tools", "END": END},
    )
    # execute_tools always feeds back into call_model
    graph.add_edge("execute_tools", "call_model")

    return graph.compile()


class GraphAgent:
    """AIAgent backed by a LangGraph compiled graph.

    Drop-in replacement for AIAgent with the same run_conversation() interface.
    The main difference is that the tool loop is now a proper state machine
    rather than an imperative for-loop, making it observable and extensible.

    Example:
        agent = GraphAgent(provider="openai", model="gpt-4o-mini")
        result = agent.run_conversation("List files in workspace")
        print(result.assistant_text)
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        toolsets: list[str] | None = None,
        isolated: bool = False,
    ) -> None:
        self.isolated = isolated
        self.sessions = SessionStore()
        self.prompt_builder = PromptBuilder()
        self.learning = LearningLoop()
        self._graph = build_agent_graph(
            provider=provider, model=model, toolsets=toolsets
        )
        self._ensure_tools_loaded()

    def _ensure_tools_loaded(self) -> None:
        import harness_agent.delegate  # noqa: F401
        import harness_agent.sessions.store  # noqa: F401
        import harness_agent.skills.loader  # noqa: F401
        import harness_agent.tools.file_tools  # noqa: F401
        import harness_agent.tools.terminal_tool  # noqa: F401

    def run_conversation(
        self,
        user_input: str,
        *,
        session_id: str | None = None,
    ) -> AgentTurnResult:
        if not session_id:
            session_id = self.sessions.create_session(title=user_input[:60])

        history = [] if self.isolated else self.sessions.load_messages(session_id)
        system_prompt = self.prompt_builder.build_system_prompt()

        initial_messages: list[Message] = [
            Message(role="system", content=system_prompt),
            *history,
            Message(role="user", content=user_input),
        ]

        initial_state: AgentState = {
            "messages": initial_messages,
            "session_id": session_id,
            "tool_call_count": 0,
            "had_error": False,
            "final_text": "",
        }

        result_state = self._graph.invoke(initial_state)

        # Extract final answer — prefer state.final_text, fall back to last message
        final_text = result_state.get("final_text", "")
        if not final_text:
            for msg in reversed(result_state["messages"]):
                if msg.role == "assistant" and msg.content and not msg.tool_calls:
                    final_text = msg.content
                    break

        # Persist new turns to session store
        if not self.isolated:
            new_msgs = result_state["messages"][len(initial_messages):]
            for msg in new_msgs:
                self.sessions.append_turn(session_id, msg)

        tool_call_count = result_state.get("tool_call_count", 0)
        had_error = result_state.get("had_error", False)

        # Learning loop — same trigger as AIAgent
        if not self.isolated and self.learning.should_author_skill(tool_call_count, had_error):
            maybe_write_skill(
                skill_name=f"graph-workflow-{session_id[:8]}",
                description=f"Auto skill from graph session with {tool_call_count} tool calls",
                body=f"User goal:\n{user_input}\n\nOutcome:\n{final_text}",
                tool_call_count=tool_call_count,
            )

        return AgentTurnResult(
            assistant_text=final_text,
            messages=result_state["messages"],
            tool_call_count=tool_call_count,
            session_id=session_id,
        )
