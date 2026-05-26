# ch04_tool_registry

# Tool registry

Harness Agent tutorial — `ch04_tool_registry.ipynb`


## Chapter objectives

By the end of this chapter you will be able to:

- Explain the **register-at-import** pattern and why it is used.
- Trace the three-step lifecycle: `register()` → `openai_schemas()` → `dispatch()`.
- Describe how **toolsets** and **check_fn** filter which tools the model sees.
- Understand the singleton `get_registry()` and why it must be module-level.
- Write and register a custom tool with a JSON Schema parameter definition.
- Predict what `dispatch()` returns when a tool raises an exception.

## Prerequisites

Prior chapters through ch04; see SYLLABUS.md.


## Concept: Tool registry

### The problem it solves

The model receives a list of JSON schemas describing what tools exist. At dispatch time,
the harness needs to call the right Python function. The **registry** is the authoritative
map between the two: `name → (schema, handler)`.

### Register-at-import pattern

Each tool module (`file_tools.py`, `terminal_tool.py`, …) calls `register()` at module
level — outside any class or function. When `AIAgent.__init__` imports the module
(`import harness_agent.tools.file_tools`) the registration happens automatically:

```python
# inside file_tools.py — runs at import time
register(
    name="read_file",
    description="Read a file from the workspace.",
    parameters={...},   # JSON Schema
    handler=read_file,  # Python callable
    toolset="files",
)
```

This pattern means: **adding a new tool = adding a new module + one `register()` call**.
No changes to `AIAgent` or any other file.

### Three-step lifecycle

```text
Step 1  REGISTER   registry._tools["read_file"] = ToolSpec(handler=read_file, ...)
Step 2  SCHEMAS    openai_schemas() → list of {type, function: {name, description, parameters}}
Step 3  DISPATCH   dispatch("read_file", {"path": "x.py"}) → handler(**args) → str
```

Step 2 happens each turn (schemas passed to `prov.complete_with_tools`).
Step 3 happens for each `ToolCall` the model returns.

### Toolsets — scoping tools

Every `ToolSpec` has a `toolset: str` (default `"default"`). When `AIAgent` is
created with `toolsets=["files", "sessions"]`, only tools in those toolsets appear
in the schema list sent to the model. This prevents the model from seeing tools it
should not use (e.g. hiding `delegate_subagent` from a subagent).

### check_fn — runtime availability

An optional `check_fn: () → bool` lets a tool opt out at runtime. If `check_fn()`
returns `False`, `list_available()` omits the tool and `dispatch()` returns an error.
Use this for tools that require optional dependencies (e.g. Docker, MCP, e2b).

## Architecture

```mermaid
flowchart LR
  subgraph import_time [At import time]
    FT[file_tools.py] -->|register()| REG[(ToolRegistry\n_tools dict)]
    TT[terminal_tool.py] -->|register()| REG
    DL[delegate.py] -->|register()| REG
  end
  subgraph turn [Each loop turn]
    REG -->|openai_schemas()| SCHEMA[tool schema list]
    SCHEMA -->|sent with messages| MODEL[LLM]
    MODEL -->|ToolCall name+args| DISPATCH[registry.dispatch]
    DISPATCH -->|handler(**args)| HANDLER[Python function]
    HANDLER -->|str result| DISPATCH
  end
```

**Key invariant**: the registry is a **singleton** (`_registry` module global in
`tools/registry.py`). Every `get_registry()` call returns the same instance, so
tools registered by one import are visible everywhere in the process.

## Reference implementation map

| Harness Agent | Nous Research agent (`REFERENCE_REPO_PATH`) | OpenClaw |
|---------------|---------------------------------------------|----------|
| ``tools/registry.py`, `tools/file_tools.py`` | search architecture guide | SOUL/gateway patterns |

Open upstream files only under your optional clone — not bundled in this tutorial.


## Design choices in harness_agent

| Choice | Rationale |
|--------|-----------|
| Module-level `register()` | Zero boilerplate — add a file, get a tool |
| String-keyed dict | O(1) dispatch; easy serialisation for debugging |
| `dispatch()` catches all exceptions | The model never sees a Python traceback — only structured JSON |
| Toolsets as strings | Lightweight; no class hierarchy needed for simple grouping |
| Singleton registry | Avoids passing a registry object through every constructor |

**Exception handling in dispatch (line 91-94 in registry.py):**

```python
try:
    return spec.handler(**arguments)
except Exception as exc:
    return wrap_exception(exc)
```

The registry delegates exception formatting to `observations.wrap_exception()`, which
produces a structured JSON the model can reason about (ch05).

## Implementation walkthrough

The cells below use the real `ToolRegistry` — no API key needed.
We inspect registered tools, call `openai_schemas()`, dispatch a real tool,
and write a custom tool from scratch.

```python
from harness_agent.tools.registry import get_registry
import harness_agent.tools.file_tools   # trigger registration
import harness_agent.tools.terminal_tool

r = get_registry()
available = r.list_available()

print(f"Registered tools: {len(available)}")
for name in available:
    spec = r._tools[name]
    print(f"  {name:30s}  toolset={spec.toolset!r}")

```

## openai_schemas() — what the model sees

Each turn, `openai_schemas()` builds the list sent to the provider.
The cell below prints one schema in full and shows toolset filtering.

```python
import json

# All schemas
all_schemas = r.openai_schemas()
print(f"Total schemas (no filter)   : {len(all_schemas)}")

# Filter to files toolset only
file_schemas = r.openai_schemas(toolsets=["files"])
print(f"Schemas with toolsets=['files']: {len(file_schemas)}\n")

# Print one schema in full so you can see what the model receives
if file_schemas:
    print("Example schema sent to model:")
    print(json.dumps(file_schemas[0], indent=2))

```

## Hands-on exercises

**Exercise 1 — Dispatch a real tool**

```python
result = r.dispatch("read_file", {"path": "nonexistent.txt"})
print(result)
```

Run this. What `status` does the observation have? What `next_actions` does it suggest?

**Exercise 2 — Write and register a custom tool**

```python
from harness_agent.tools.registry import register
import json

def word_count(text: str) -> str:
    from harness_agent.observations import wrap_result
    count = len(text.split())
    return wrap_result(status="success", summary=f"{count} words",
                       detail=str(count))

register(
    name="word_count",
    description="Count the words in a text string.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to count"}},
        "required": ["text"],
    },
    handler=word_count,
    toolset="demo",
)

print(r.dispatch("word_count", {"text": "hello world this is harness agent"}))
print("Available after registration:", "word_count" in r.list_available())
```

**Exercise 3 — check_fn**

Add a `check_fn` to your `word_count` registration that returns `False` when
`os.environ.get("DISABLE_WORD_COUNT")` is set. Verify `list_available()` omits it.

## Common pitfalls

| Pitfall | Root cause | Fix |
|---------|-----------|-----|
| Tool not appearing in schemas | Module not imported before `openai_schemas()` | Import the module in `AIAgent._ensure_tools_loaded()` |
| `dispatch()` returns error for valid name | `check_fn()` returns `False` | Check the availability condition |
| Duplicate registration | Two modules call `register()` with the same name | Second registration silently overwrites first — use unique names |
| Arguments type mismatch | Model passes string, handler expects int | Add type coercion in handler or use JSON Schema `type` |
| Handler returns non-string | `dispatch()` expects a `str` to append as tool message | Return `json.dumps(...)` or use `wrap_result()` |
| Forgetting `required` in parameters | Optional field treated as required by model | Always list required fields explicitly |

## Checkpoint questions

1. **Register-at-import** — Why does `file_tools.py` call `register()` at module level rather than inside a function? What would break if you moved it inside `main()`?

2. **Singleton** — What happens if two different parts of the codebase call `get_registry()` independently? Do they get the same object or different instances?

3. **Toolset filtering** — A subagent is created with `AIAgent(isolated=True, toolsets=["files"])`. Which tools will `openai_schemas()` return? Which tool will be absent that the parent agent sees?

4. **dispatch exceptions** — What does `dispatch()` return when the handler raises a `FileNotFoundError`? Which function produces this output?

5. **openai_schemas format** — Write out the Python dict structure for a tool schema entry. What are the top-level keys? What's inside `"function"`?

6. **check_fn** — Give a concrete example of when you would set `check_fn` on a tool registration. What does `list_available()` do when `check_fn()` returns `False`?

## Summary & next chapter

| Topic | Key takeaway |
|-------|-------------|
| Register-at-import | Tools self-register when their module is imported; no central list to maintain |
| `ToolSpec` | Stores `name`, `description`, `parameters` (JSON Schema), `handler`, `toolset`, `check_fn` |
| `openai_schemas()` | Builds the model's tool list; filtered by toolset and check_fn |
| `dispatch()` | Routes by name, catches all exceptions, returns structured JSON string |
| Singleton registry | One global `_registry` instance shared across the entire process |
| Exception safety | `dispatch()` never propagates Python exceptions to the agent loop |

**ch05** covers **observations and recovery** — the structured JSON format that every
tool must return so the model can reason about success, warning, and error states.
