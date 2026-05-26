# ch13_mcp_integration

# MCP integration

Harness Agent tutorial — `ch13_mcp_integration.ipynb`


## Chapter objectives

By the end of this chapter you will be able to:

- Explain what MCP (Model Context Protocol) is and why it extends the tool registry.
- Trace `MCPIntegration.connect_stdio()` from server launch through tool registration.
- Describe how MCP tools are named (`mcp_*`) and registered in the harness registry.
- Explain the async-to-sync wrapper and why `asyncio.run` is used.
- Connect a real MCP server (if available) and verify tools appear in `list_available()`.
- Understand what happens when the `mcp` package is not installed.

## Prerequisites

Prior chapters through ch13; see SYLLABUS.md.


## Concept: MCP integration

### What is MCP?

The **Model Context Protocol** (MCP, by Anthropic) is a standard for connecting
AI assistants to external tools via a JSON-RPC server. Instead of writing a Python
handler in the harness codebase, you run an MCP server that exposes any number of
tools over stdio or HTTP.

This allows:
- Reusing the same MCP server with multiple harnesses.
- Adding tools without modifying harness code.
- Using tools in other languages (Node.js, Go, Rust, …).

### How MCPIntegration works

```text
connect_stdio(command, args)
   │
   ├── launch MCP server subprocess (stdio_client)
   ├── session.initialize()
   ├── tools = session.list_tools()
   │
   └── for each tool:
         name = f"mcp_{tool.name}"
         async_handler = lambda **kwargs: sess.call_tool(name, kwargs)
         sync_handler = lambda **kwargs: asyncio.run(async_handler(**kwargs))
         registry.register(name, description, parameters, sync_handler, toolset="mcp")
```

### Async-to-sync bridge

MCP clients are async (`async with stdio_client`). The harness registry is synchronous.
`MCPIntegration` wraps each async handler in `asyncio.run()` to bridge them:

```python
def sync_handler(**kwargs: Any, _ah=async_handler) -> str:
    return asyncio.run(_ah(**kwargs))
```

**Limitation**: `asyncio.run()` cannot be called from inside a running event loop.
If your harness is async, you need a different bridge (e.g. `nest_asyncio`).

### Tool naming convention

MCP tools are prefixed with `mcp_` to avoid naming collisions with native tools.
A server with a `read_file` tool becomes `mcp_read_file` in the registry.

### Graceful degradation

```python
try:
    from mcp import ClientSession, StdioServerParameters
except ImportError as exc:
    raise RuntimeError("Install mcp package to use MCP integration") from exc
```

If the `mcp` package is not installed, the `connect_stdio` method raises a clear
`RuntimeError` rather than an `ImportError`.

## How it works

`MCPIntegration.connect_stdio_sync` registers `mcp_*` tools.

```mermaid
flowchart LR
  U[User or scheduler] --> A[AIAgent]
  A --> M[MCP integration]
```

Trace cells below execute real code paths offline where possible.


## Reference implementation map

| Harness Agent | Nous Research agent (`REFERENCE_REPO_PATH`) | OpenClaw |
|---------------|---------------------------------------------|----------|
| ``mcp/client.py`` | search architecture guide | SOUL/gateway patterns |

Open upstream files only under your optional clone — not bundled in this tutorial.


## Design choices in harness_agent

Tutorial implementation prioritizes readable Python over feature parity. Extend ``mcp/client.py`` as exercises.


## Implementation walkthrough


```python
import inspect
from harness_agent.mcp.client import MCPIntegration

# Inspect the class
print(f"MCPIntegration docstring: {MCPIntegration.__doc__!r}")
print()

# Inspect connect_stdio signature
sig = inspect.signature(MCPIntegration.connect_stdio)
print(f"connect_stdio signature: {sig}")
print()

# Inspect connect_stdio_sync
sig2 = inspect.signature(MCPIntegration.connect_stdio_sync)
print(f"connect_stdio_sync signature: {sig2}")
print()

# Show the source of the sync bridge
src = inspect.getsource(MCPIntegration.connect_stdio_sync)
print("connect_stdio_sync source:")
for i, line in enumerate(src.splitlines(), 1):
    print(f"  {i:2d}  {line}")

print()
m = MCPIntegration()
print(f"_attached before any connection: {m._attached}")
```

## Trace one request


```python
from harness_agent.tools.registry import get_registry

r = get_registry()
mcp_tools = [t for t in r.list_available() if t.startswith("mcp_")]

if mcp_tools:
    print(f"MCP tools registered: {mcp_tools}")
else:
    print("No MCP tools registered (no server connected yet)")
    print()
    print("To connect a server:")
    print("  mcp_int = MCPIntegration()")
    print("  registered = mcp_int.connect_stdio_sync('uvx', ['mcp-server-filesystem', '.'])")
    print("  print('Registered:', registered)")
    print()
    print("Example MCP servers available via uvx:")
    print("  mcp-server-filesystem  — file system operations")
    print("  mcp-server-git         — git repository operations")
    print("  mcp-server-sqlite      — SQLite database access")

# Demonstrate graceful ImportError
import sys
print()
print("Graceful degradation when mcp not installed:")
try:
    from harness_agent.mcp.client import MCPIntegration as M
    mcp_inst = M()
    # This will only fail if we actually call connect_stdio and mcp isn't installed
    print("  MCPIntegration class imported successfully")
    print("  RuntimeError raised only when connect_stdio() is called without mcp package")
except Exception as e:
    print(f"  Error: {e}")
```

## Hands-on exercises

**Exercise 1 — Connect a real MCP server (requires mcp package)**

```bash
pip install mcp
uvx mcp-server-filesystem . &  # run in terminal
```

Then in the notebook:

```python
from harness_agent.mcp.client import MCPIntegration
mcp = MCPIntegration()
tools = mcp.connect_stdio_sync("uvx", ["mcp-server-filesystem", "."])
print("Registered:", tools)

r = get_registry()
print("mcp_* in registry:", [t for t in r.list_available() if t.startswith("mcp_")])
```

**Exercise 2 — Dispatch an MCP tool**

After connecting a filesystem server:

```python
result = r.dispatch("mcp_list_directory", {"path": "."})
print(result)
```

Verify it returns a structured observation.

**Exercise 3 — Tool naming collision**

What happens if an MCP server exposes a tool called `read_file`? How does the
`mcp_` prefix prevent it from colliding with the native `read_file` tool?

**Exercise 4 — Async limitation**

Try calling `connect_stdio_sync` from inside a Jupyter cell that already has an
event loop running. What error do you get? How would you fix it with `nest_asyncio`?

```python
import nest_asyncio
nest_asyncio.apply()
```

## Common pitfalls

| Pitfall | Root cause | Fix |
|---------|-----------|-----|
| `RuntimeError: Install mcp package` | `mcp` not installed | `pip install mcp` |
| `asyncio.run()` nested loop error | Calling from Jupyter (already has loop) | `pip install nest_asyncio; nest_asyncio.apply()` |
| MCP tools not persisting after restart | Registry is in-memory | Call `connect_stdio_sync` again each session |
| Tool naming collision | Server uses same name as native tool | MCP prefix `mcp_` prevents this — don't remove it |
| Server process not starting | Wrong command or path | Test with `subprocess.run([command] + args)` first |
| Result not structured | MCP server returns plain text | Wrapped in `wrap_result(status="success", detail=text)` |

## Checkpoint questions

1. **MCP protocol** — What transport does `connect_stdio()` use? What two MCP SDK classes does it import?

2. **Tool naming** — An MCP server exposes a tool `search_web`. What is its name in the Harness Agent registry after connection?

3. **Async bridge** — `sync_handler` calls `asyncio.run(_ah(**kwargs))`. When does this fail? What is the workaround?

4. **_attached list** — After `connect_stdio_sync`, where are the registered tool names tracked? How would you disconnect/de-register them?

5. **Graceful degradation** — If `mcp` is not installed, when exactly does the `RuntimeError` fire — at import time or at connection time?

6. **Toolset** — MCP tools are registered with `toolset="mcp"`. What does this mean for `AIAgent(toolsets=["files"])`? Will MCP tools be available?

## Summary & next chapter

| Topic | Key takeaway |
|-------|-------------|
| MCP | External tool server standard; extends harness without code changes |
| `connect_stdio_sync()` | Launches server, lists tools, registers them as `mcp_*` in registry |
| Async bridge | `asyncio.run(async_handler)` — fails if event loop already running |
| Tool prefix | `mcp_` prevents naming collisions with native tools |
| `toolset="mcp"` | MCP tools only available when `toolsets` includes `"mcp"` or is `None` |
| Graceful degradation | Import succeeds; `RuntimeError` raised only on first `connect_stdio` call |

**ch14** covers the **cron scheduler** — how agent jobs are defined in JSON and
executed on a time-based schedule by `CronScheduler.tick()`.
