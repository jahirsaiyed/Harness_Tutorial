# ch22_langgraph_deepagents_sandbox

# LangGraph, DeepAgents & Sandbox

Harness Agent tutorial — `ch22_langgraph_deepagents_sandbox.ipynb`

## Chapter objectives

- Replace the manual `for` loop in `AIAgent` with a **LangGraph state machine**.
- Understand `AgentState`, nodes (`call_model`, `execute_tools`), and conditional edges.
- Build and run a **multi-agent supervisor graph** (DeepAgents pattern): supervisor routes tasks to specialist workers.
- Use the new **sandbox abstraction** (`local` → `docker` → `e2b`) to run code safely.
- Stream intermediate state updates from both graph types.

## Prerequisites

```bash
pip install -e ".[dev]"
# For e2b cloud sandbox (optional):
pip install -e ".[sandbox]"
```

API key for your preferred provider in `.env`.

## Part 1 — Why LangGraph?

The original `AIAgent.run_conversation()` uses a plain for-loop:

```python
for _ in range(self.config.max_turns):
    text, calls = prov.complete_with_tools(...)
    if calls:
        # dispatch tools, continue
    else:
        break  # done
```

This works, but it has three limitations:

| Limitation | Impact |
|-----------|--------|
| Opaque — no visibility into intermediate states | Hard to debug multi-turn tool chains |
| Rigid — inserting a compression or guard step means editing the loop | Fragile to maintain |
| Not streamable — result only available at the end | No real-time feedback |

**LangGraph** models the same loop as a directed graph of nodes and edges:

```
[START]
   │
   ▼
call_model ──(tool_calls?)──> execute_tools ──┐
   │                                          │
   │ (no tool_calls)                          │
   ▼                                          │
 [END]  <────────────────────────────────────┘
```

Benefits:
- Every transition is observable via `.stream()`
- Nodes are plain functions — easy to unit-test
- Adding a new step (guard, compression, logging) = adding a node + edge

## Part 2 — LangGraph core concepts

### State

`AgentState` is a `TypedDict`. Every node receives the current state and returns a **partial update** — a dict with only the keys it changes. LangGraph merges updates using **reducers**.

```python
class AgentState(TypedDict):
    messages: Annotated[list[Message], _append_messages]  # reducer: list append
    session_id: str | None                                # reducer: overwrite
    tool_call_count: int
    had_error: bool
    final_text: str
```

### Nodes

A node is any `(state) -> dict` function:

```python
def call_model(state: AgentState) -> dict:
    text, calls = provider.complete_with_tools(state["messages"], schemas)
    return {"messages": [Message(role="assistant", content=text, tool_calls=...)]}  
    # list append reducer adds this to state["messages"]
```

### Conditional edges

After `call_model`, a router function inspects the state and returns a string:

```python
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.role == "assistant" and last.tool_calls:
        return "execute_tools"   # → run the tools
    return "END"                 # → terminate
```

```python
import os
from pathlib import Path

os.environ.setdefault('HARNESS_AGENT_HOME', str(Path('labs').resolve()))

# Verify langgraph is installed
import langgraph
print(f"langgraph version: {langgraph.__version__}")

from harness_agent.graph.state import AgentState, SupervisorState
from harness_agent.graph.nodes import should_continue, make_call_model_node, make_execute_tools_node
from harness_agent.graph.agent_graph import build_agent_graph, GraphAgent
from harness_agent.graph.multi_agent_graph import build_multi_agent_graph, MultiAgentOrchestrator
from harness_agent.sandbox.base import get_sandbox
print("All imports OK")
```

## Part 3 — Inspect the agent graph structure

```python
# Build the graph and inspect its structure without running it
graph = build_agent_graph(model="gpt-4o-mini")

print("=== Graph nodes ===")
for node in graph.nodes:
    print(f"  {node}")

print()
print("=== Graph edges ===")
for edge in graph.edges:
    print(f"  {edge}")
```

```python
# Visualise the graph (requires graphviz — skips gracefully if not installed)
try:
    from IPython.display import Image, display
    img_data = graph.get_graph().draw_mermaid_png()
    display(Image(img_data))
except Exception as e:
    print(f"Visualisation skipped: {e}")
    # Mermaid diagram as text fallback:
    print(graph.get_graph().draw_mermaid())
```

## Part 4 — Run GraphAgent (single-agent graph)

`GraphAgent` is a drop-in replacement for `AIAgent` with the same `run_conversation()` interface.

```python
agent = GraphAgent(model="gpt-4o-mini", toolsets=["files"])

result = agent.run_conversation("List all files in the workspace and summarise what you find.")

print("=== Answer ===")
print(result.assistant_text)
print()
print(f"Tool calls made : {result.tool_call_count}")
print(f"Session ID      : {result.session_id}")
print(f"Total messages  : {len(result.messages)}")
```

## Part 5 — Stream intermediate state updates

Unlike the original for-loop, the graph can emit each node's output as it runs.
This lets you watch the tool loop in real time.

```python
from harness_agent.types import Message
from harness_agent.graph.state import AgentState
from harness_agent.prompt.builder import PromptBuilder

graph = build_agent_graph(model="gpt-4o-mini", toolsets=["files"])

system_prompt = PromptBuilder().build_system_prompt()
initial_state: AgentState = {
    "messages": [
        Message(role="system", content=system_prompt),
        Message(role="user", content="Write hello_world.py to the workspace, then read it back."),
    ],
    "session_id": None,
    "tool_call_count": 0,
    "had_error": False,
    "final_text": "",
}

print("=== Streaming graph execution ===")
for step in graph.stream(initial_state):
    node_name = list(step.keys())[0]
    node_output = step[node_name]
    new_msgs = node_output.get("messages", [])
    print(f"\n[{node_name}]")
    for msg in new_msgs:
        role = msg.role
        snippet = (msg.content or "")[:120].replace("\n", " ")
        tool_info = f" | tool_calls: {len(msg.tool_calls)}" if msg.tool_calls else ""
        print(f"  {role}: {snippet}{tool_info}")
```

## Part 6 — The Sandbox

The `run_shell` tool now routes through `get_sandbox()` instead of `get_terminal_backend()` directly.

### Three backends

| Backend | Isolation | When to use | Activation |
|---------|-----------|-------------|------------|
| `local` | None — subprocess in workspace | Development, trusted code | default |
| `docker` | Strong — ephemeral container, `--network none`, memory cap | Untrusted commands | `HARNESS_SANDBOX=docker` |
| `e2b` | Strongest — cloud Firecracker microVM | User-submitted code, CI | `HARNESS_SANDBOX=e2b` |

All three expose the same interface:

```python
sandbox.run(command)            # → (exit_code, stdout, stderr)
sandbox.write_file(path, text)  # write into sandbox
sandbox.read_file(path)         # read back from sandbox
```

```python
from harness_agent.sandbox.base import get_sandbox

# Show which backend is active
sandbox = get_sandbox()
print(f"Active sandbox backend: {sandbox.name}")
print()

# Write a file and run it
sandbox.write_file("sandbox_demo.py", """
import sys, platform
print(f"Python {sys.version}")
print(f"Platform: {platform.system()}")
print("Sandbox execution OK")
""")

code, out, err = sandbox.run("python sandbox_demo.py")
print(f"Exit code : {code}")
print(f"stdout    :\n{out}")
if err:
    print(f"stderr    :\n{err}")
```

```python
# Demonstrate all three backends (falls back gracefully when unavailable)
from harness_agent.sandbox.local_sandbox import LocalSandbox
from harness_agent.sandbox.docker_sandbox import DockerSandbox, docker_available

backends_to_test = [("local", LocalSandbox())]
if docker_available():
    backends_to_test.append(("docker", DockerSandbox()))
else:
    print("Docker not available — skipping docker backend test")

for name, sb in backends_to_test:
    print(f"\n=== {name} sandbox ===")
    code, out, err = sb.run("echo 'hello from sandbox' && python3 -c 'print(1+1)'")
    print(f"  exit={code}  stdout={out.strip()!r}")
```

## Part 7 — Coder agent with Docker sandbox

Wire the `coder` toolset to the Docker sandbox so code runs in a container.

```python
import os

# Switch to docker sandbox for this cell (no-op if Docker is unavailable)
if docker_available():
    os.environ["HARNESS_SANDBOX"] = "docker"
    print("Sandbox: docker (isolated container)")
else:
    os.environ["HARNESS_SANDBOX"] = "local"
    print("Sandbox: local (Docker not available)")

# GraphAgent with terminal toolset — run_shell now uses the selected sandbox
coder_agent = GraphAgent(model="gpt-4o-mini", toolsets=["terminal", "files"])

result = coder_agent.run_conversation(
    "Write a Python script called fizzbuzz.py that prints FizzBuzz for 1-20, "
    "save it to the workspace, then run it and show me the output."
)

print("=== Coder agent answer ===")
print(result.assistant_text)
print(f"\nTool calls: {result.tool_call_count}")

# Reset sandbox to local
os.environ["HARNESS_SANDBOX"] = "local"
```

## Part 8 — Multi-agent supervisor (DeepAgents pattern)

The supervisor graph models the **deep research pattern**: a coordinator dispatches
sequential sub-tasks to specialised workers and aggregates their findings.

```
[START]
   │
   ▼
supervisor ──> researcher ──┐
           ├─> coder      ──┤──> supervisor (loop)
           ├─> planner    ──┘
           └─> FINISH ──> [END]
```

### Worker specialisations

| Worker | Toolset | Role |
|--------|---------|------|
| `researcher` | files, sessions | Reads files, searches history, gathers information |
| `coder` | terminal, files | Writes and runs code in the sandbox |
| `planner` | none | Pure reasoning — task decomposition, planning |

```python
# Inspect the multi-agent graph structure
multi_graph = build_multi_agent_graph(model="gpt-4o-mini")

print("=== Multi-agent graph nodes ===")
for node in multi_graph.nodes:
    print(f"  {node}")

try:
    from IPython.display import Image, display
    display(Image(multi_graph.get_graph().draw_mermaid_png()))
except Exception:
    print(multi_graph.get_graph().draw_mermaid())
```

```python
orchestrator = MultiAgentOrchestrator(model="gpt-4o-mini")

task = (
    "First, write a file called data.txt with the numbers 1 to 10 (one per line). "
    "Then write a Python script that reads data.txt and prints the sum. "
    "Run the script and report the result."
)

print(f"Task: {task}")
print("\n=== Orchestrating...")
answer = orchestrator.run(task)
print("\n=== Worker summaries ===")
print(answer)
```

```python
# Stream the supervisor/worker dialogue in real time
print("=== Streaming multi-agent execution ===")

for step in orchestrator.stream("Describe what files exist in the workspace."):
    node_name = list(step.keys())[0]
    node_output = step[node_name]
    print(f"\n[{node_name}]")

    if "next_worker" in node_output:
        print(f"  → next worker: {node_output['next_worker']}")
    if "worker_results" in node_output and node_output["worker_results"]:
        for r in node_output["worker_results"]:
            print(f"  result: {r[:200]}")
```

## Part 9 — e2b cloud sandbox (optional)

The `e2b` backend runs code in a fully isolated Firecracker microVM on e2b.dev infrastructure.

```bash
pip install e2b-code-interpreter
# Set in .env:
E2B_API_KEY=your_key_here
```

```python
import os

e2b_key = os.environ.get("E2B_API_KEY", "")
if not e2b_key:
    print("E2B_API_KEY not set — skipping e2b demo.")
    print("Set it in .env to run code in a cloud microVM.")
else:
    os.environ["HARNESS_SANDBOX"] = "e2b"
    try:
        from harness_agent.sandbox.e2b_sandbox import E2BSandbox
        sbx = E2BSandbox()
        code, out, err = sbx.run("python3 -c 'import platform; print(platform.node())'")
        print(f"e2b sandbox hostname: {out.strip()}")
        print("(This is a remote microVM, not your local machine)")
    except ImportError as e:
        print(f"e2b not installed: {e}")
    finally:
        os.environ["HARNESS_SANDBOX"] = "local"
```

## Part 10 — Compare AIAgent vs GraphAgent

Both produce identical outputs. The difference is internal structure.

```python
from harness_agent.agent import AIAgent

query = "What files are in the workspace?"

# Original agent
original = AIAgent(isolated=True, toolsets=["files"])
r1 = original.run_conversation(query)

# LangGraph agent
graph_ag = GraphAgent(isolated=True, toolsets=["files"], model="gpt-4o-mini")
r2 = graph_ag.run_conversation(query)

print("=== AIAgent (for-loop) ===")
print(r1.assistant_text[:300])
print()
print("=== GraphAgent (LangGraph) ===")
print(r2.assistant_text[:300])
print()
print(f"AIAgent   tool calls: {r1.tool_call_count}")
print(f"GraphAgent tool calls: {r2.tool_call_count}")
```

## Hands-on exercises

1. **Add a guard node**: insert a `validate_output` node between `execute_tools` and `call_model` that logs tool results to a file. Add it to `build_agent_graph` with `graph.add_node` + `graph.add_edge`.

2. **Add a worker**: create a `summariser` worker in `nodes.py` that reads all `.txt` files in the workspace and returns a summary. Add it to `build_multi_agent_graph`.

3. **Switch sandboxes at runtime**: modify the coder agent demo (Part 7) to run one command in `local` and one in `docker`, comparing their outputs.

4. **Persist graph state**: LangGraph supports checkpointers (`MemorySaver`, `SqliteSaver`). Add `from langgraph.checkpoint.sqlite import SqliteSaver` and pass `checkpointer=saver` to `graph.compile()`. This enables resuming a graph mid-execution across process restarts.

## Common pitfalls

| Pitfall | Fix |
|---------|-----|
| `langgraph` not installed | `pip install langgraph>=0.2.0` |
| State not updating — node returns full state instead of partial | Return only the keys your node changes; reducers merge them |
| Messages duplicated in state | The `_append_messages` reducer appends; don't include prior messages in the return dict |
| Supervisor loops infinitely | `_turn_guard` enforces `MAX_SUPERVISOR_TURNS=10`; lower it for debugging |
| Docker sandbox times out | Increase `timeout` kwarg or use `--memory` / `--cpus` flags in `DockerSandbox.run()` |
| e2b sandbox not found | `pip install e2b-code-interpreter` and set `E2B_API_KEY` |
| `HARNESS_SANDBOX` has no effect | It is read fresh on each `get_sandbox()` call; set it before tool dispatch |

## Summary

### What was added to the project

| Module | What it does |
|--------|--------------|
| `harness_agent/sandbox/base.py` | `SandboxBackend` ABC + `get_sandbox()` factory |
| `harness_agent/sandbox/local_sandbox.py` | Subprocess backend (dev/testing) |
| `harness_agent/sandbox/docker_sandbox.py` | Ephemeral container, network-isolated |
| `harness_agent/sandbox/e2b_sandbox.py` | Cloud microVM via e2b.dev |
| `harness_agent/graph/state.py` | `AgentState` + `SupervisorState` TypedDicts with reducers |
| `harness_agent/graph/nodes.py` | `call_model`, `execute_tools`, `should_continue`, `supervisor`, `worker` nodes |
| `harness_agent/graph/agent_graph.py` | `GraphAgent` + `build_agent_graph()` |
| `harness_agent/graph/multi_agent_graph.py` | `MultiAgentOrchestrator` + `build_multi_agent_graph()` |
| `tools/terminal_tool.py` (updated) | `run_shell` now routes through `get_sandbox()` |

### Key takeaways

- **LangGraph** replaces the manual for-loop with a state machine: same behaviour, fully observable and extensible.
- **DeepAgents pattern** = supervisor LLM that routes sub-tasks to specialist workers; workers each run their own mini tool loop.
- **Sandbox layer** unifies local/Docker/e2b behind one interface; select via `HARNESS_SANDBOX` env var.
- `GraphAgent` and `AIAgent` are interchangeable — same `run_conversation()` API.
