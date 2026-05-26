"""Multi-agent supervisor graph — DeepAgents pattern.

Architecture (supervisor + specialist workers):

    [START]
       │
       ▼
  supervisor ──> researcher ──┐
             ├─> coder      ──┤──> supervisor (loop)
             ├─> planner    ──┘
             └─> FINISH ──> [END]

How it works
------------
1. The supervisor LLM sees: the original task + accumulated worker_results.
2. It picks one of: researcher | coder | planner | FINISH.
3. The chosen worker runs a focused mini tool-loop (max 5 turns) with
   its allowed toolset, then appends a summary to worker_results.
4. Control returns to the supervisor, which reassesses and routes again.
5. When the supervisor decides FINISH (or the safety turn limit is hit),
   the graph terminates and returns the aggregated worker_results.

Worker specialisations
-----------------------
  researcher  files + sessions toolsets — reads files, searches history
  coder       terminal + files toolsets — runs code in the sandbox
  planner     no tools           — pure reasoning, task decomposition

This mirrors the "deep research" pattern where a coordinator dispatches
parallel or sequential sub-tasks to specialised agents and aggregates results.

Example:
    orchestrator = MultiAgentOrchestrator(provider="openai", model="gpt-4o-mini")
    answer = orchestrator.run("Analyse logs/app.log and produce a fix plan")
    print(answer)
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from harness_agent.graph.nodes import (
    WORKERS,
    make_supervisor_node,
    make_worker_node,
)
from harness_agent.graph.state import SupervisorState

MAX_SUPERVISOR_TURNS = 10


def _turn_guard(state: SupervisorState) -> str:
    """Safety edge: force FINISH if turn limit exceeded."""
    if state.get("turn", 0) >= MAX_SUPERVISOR_TURNS:
        return "FINISH"
    return state.get("next_worker", "FINISH")


def build_multi_agent_graph(
    *,
    provider: str | None = None,
    model: str | None = None,
) -> Any:
    """Compile and return a runnable multi-agent supervisor graph.

    Args:
        provider:  Provider name. Falls back to HARNESS_DEFAULT_PROVIDER.
        model:     Model identifier. Falls back to HARNESS_DEFAULT_MODEL.

    Returns:
        A compiled LangGraph graph accepting SupervisorState.
    """
    supervisor_node = make_supervisor_node(provider=provider, model=model)
    researcher_node = make_worker_node("researcher", provider=provider, model=model)
    coder_node = make_worker_node("coder", provider=provider, model=model)
    planner_node = make_worker_node("planner", provider=provider, model=model)

    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("coder", coder_node)
    graph.add_node("planner", planner_node)

    graph.set_entry_point("supervisor")

    # Supervisor routes to workers or terminates
    graph.add_conditional_edges(
        "supervisor",
        _turn_guard,
        {
            "researcher": "researcher",
            "coder": "coder",
            "planner": "planner",
            "FINISH": END,
        },
    )

    # All workers report back to supervisor after completing their sub-task
    for worker in WORKERS:
        graph.add_edge(worker, "supervisor")

    return graph.compile()


class MultiAgentOrchestrator:
    """High-level interface for the multi-agent supervisor graph.

    Wraps build_multi_agent_graph() with a simple run(task) → str interface.

    Example:
        orc = MultiAgentOrchestrator(provider="openai", model="gpt-4o")
        result = orc.run(
            "Read the error in logs/app.log, write a fix in fix.py, "
            "run it, and give me a summary."
        )
        print(result)
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self._graph = build_multi_agent_graph(provider=provider, model=model)

    def run(self, task: str, *, session_id: str | None = None) -> str:
        """Execute the multi-agent graph for a given task.

        Returns a string containing each worker's summary, one per line.
        """
        initial: SupervisorState = {
            "messages": [],
            "session_id": session_id,
            "task": task,
            "next_worker": "",
            "worker_results": [],
            "final_text": "",
            "turn": 0,
        }
        result_state = self._graph.invoke(initial)
        results = result_state.get("worker_results") or []
        return "\n".join(results) if results else "(no output)"

    def stream(self, task: str):
        """Yield intermediate state updates as the graph executes.

        Useful for watching the supervisor/worker dialogue in real time.
        """
        initial: SupervisorState = {
            "messages": [],
            "session_id": None,
            "task": task,
            "next_worker": "",
            "worker_results": [],
            "final_text": "",
            "turn": 0,
        }
        yield from self._graph.stream(initial)
