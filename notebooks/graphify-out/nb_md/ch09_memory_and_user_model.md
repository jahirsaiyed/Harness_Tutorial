# ch09_memory_and_user_model

# Memory and user model

Harness Agent tutorial — `ch09_memory_and_user_model.ipynb`


## Chapter objectives

By the end of this chapter you will be able to:

- Distinguish `MEMORY.md` (environmental facts) from `USER.md` (user preferences).
- Locate both files using `HarnessConfig.memory_path` and `HarnessConfig.user_path`.
- Call `load_memory_block()` and explain the two-section output format.
- Write content to both files and verify it appears in the assembled system prompt.
- Explain why memory files are injected at the system prompt level, not as chat messages.
- Design effective MEMORY.md and USER.md content for a real project.

## Prerequisites

Prior chapters through ch09; see SYLLABUS.md.


## Concept: Memory and user model

### Two types of persistent context

| File | Purpose | Who updates it |
|------|---------|---------------|
| `MEMORY.md` | **Environment facts**: what tools are installed, what services are running, project state | Agent (via `write_file` tool) or human |
| `USER.md` | **User preferences**: timezone, preferred language, communication style, project role | Human or learning loop |

Both files live under `<HARNESS_AGENT_HOME>/memory/`. Neither is a session — they
persist across all conversations.

### How they're injected

`load_memory_block()` reads both files and returns them as a combined string:

```text
# Long-term memory (MEMORY)
<contents of MEMORY.md>

# User model (USER)
<contents of USER.md>
```

This string is inserted into the system prompt by `PromptBuilder.build_system_prompt()`.
If a file doesn't exist, its section is omitted.

### Why system prompt, not chat history?

Injecting at the system level means:
- The context is available from turn 1 (no need to "say it again").
- Compression doesn't discard it (head[:2] preserved).
- The information is stable across all turns (not overwritten by new messages).

### Update pattern

The agent itself can update memory mid-session using the `write_file` tool:

```python
# Model calls this to save a fact
write_file("memory/MEMORY.md", "## Environment\n- Redis 7.2 installed\n- Port 6379\n")
```

This makes memory **write-back** capable — the agent learns environmental facts
as it works.

## How it works

Read files from `HARNESS_AGENT_HOME/memory/`; size limits in production.

```mermaid
flowchart LR
  U[User or scheduler] --> A[AIAgent]
  A --> M[Memory and user model]
```

Trace cells below execute real code paths offline where possible.


## Reference implementation map

| Harness Agent | Nous Research agent (`REFERENCE_REPO_PATH`) | OpenClaw |
|---------------|---------------------------------------------|----------|
| ``memory/files.py`` | search architecture guide | SOUL/gateway patterns |

Open upstream files only under your optional clone — not bundled in this tutorial.


## Design choices in harness_agent

Tutorial implementation prioritizes readable Python over feature parity. Extend ``memory/files.py`` as exercises.


## Implementation walkthrough


```python
from harness_agent.memory.files import load_memory_block
from harness_agent.config import get_config
from pathlib import Path

config = get_config()

print(f"memory_path : {config.memory_path}")
print(f"user_path   : {config.user_path}")
print()

# Without any files
block_empty = load_memory_block()
print(f"=== load_memory_block() with no files ===")
print(repr(block_empty) if block_empty else "(empty string — no files exist)")
print()

# Write MEMORY.md
config.memory_path.write_text("""## Environment
- Python 3.12 (venv in .venv/)
- PostgreSQL 16 on localhost:5432
- Redis 7.2 on localhost:6379

## Project state
- Main service: running
- DB migrations: up to date
""", encoding="utf-8")

# Write USER.md
config.user_path.write_text("""## Preferences
- Prefer short, direct answers
- Use British English spelling
- Timezone: Europe/London

## Role
- Backend engineer, 5 years Python
""", encoding="utf-8")

block = load_memory_block()
print("=== load_memory_block() with both files ===")
print(block)
```

## Trace one request


```python
from harness_agent.prompt.builder import PromptBuilder

# Verify memory appears in the assembled system prompt
prompt = PromptBuilder().build_system_prompt()
sections = prompt.split("\n\n")

print(f"System prompt sections: {len(sections)}")
for i, s in enumerate(sections):
    label = s.split("\n")[0][:60]
    print(f"  [{i}] {label!r}")

print()
# Find the memory section
for s in sections:
    if "Long-term memory" in s:
        print("=== MEMORY section in prompt ===")
        print(s[:300])
        break

# Clean up demo files
config.memory_path.unlink(missing_ok=True)
config.user_path.unlink(missing_ok=True)
print("\n(Demo files cleaned up)")
```

## Hands-on exercises

**Exercise 1 — Write useful MEMORY.md**

Create a `MEMORY.md` that describes your actual development environment:
- OS and Python version
- Any databases or services running
- Key directories

Run `load_memory_block()` and verify it appears correctly.

**Exercise 2 — Memory update via tool**

With a live agent (API key required), ask it to update MEMORY.md:

```
"Remember that I prefer to use pytest for testing and that the test suite
lives in tests/. Update MEMORY.md with this information."
```

After the conversation, open `MEMORY.md` — did the agent update it?

**Exercise 3 — Section headers matter**

What happens if you use `# User model (USER)` as a section header in `MEMORY.md`?
Could this confuse the agent? How would you prevent header collisions?

**Exercise 4 — Memory size limit**

Using `estimate_tokens()`, measure the token cost of a 500-line MEMORY.md.
At what point does it consume a significant fraction of the context window?
How would you design a memory rotation strategy?

## Common pitfalls

| Pitfall | Root cause | Fix |
|---------|-----------|-----|
| Memory not appearing in prompt | File at wrong path | Check `config.memory_path` — must be under `HARNESS_AGENT_HOME/memory/` |
| Memory section missing | File exists but is empty | Write at least one line |
| Memory overwritten by agent | Agent uses `write_file` with wrong path | Include safe-write instructions in SOUL.md |
| MEMORY.md too large | Agent appends without limit | Cap at ~2 000 chars; rotate old entries |
| USER.md and MEMORY.md confused | Wrong content in each file | MEMORY = environment facts; USER = human preferences |
| Memory not updated between sessions | Agent doesn't write it back | Explicitly instruct the agent to update MEMORY.md after learning new facts |

## Checkpoint questions

1. **Two files** — What is the conceptual difference between `MEMORY.md` and `USER.md`? Give a concrete example of content that belongs in each.

2. **load_memory_block** — What does the function return when both files exist? When neither exists? When only `MEMORY.md` exists?

3. **Prompt position** — In `build_system_prompt()`, does MEMORY appear before or after skill metadata? Why does this order matter?

4. **Write-back** — The agent wants to remember that Redis is now on port 6380. Which tool would it call? What path argument would it use?

5. **Compression safety** — Memory is in the system message (index 0). What does the compression algorithm do with `messages[:2]`? Why does this matter for memory persistence?

6. **Token cost** — A MEMORY.md has 3 000 characters. Approximately how many tokens does it add to the system prompt?

## Summary & next chapter

| Topic | Key takeaway |
|-------|-------------|
| `MEMORY.md` | Environment facts: tools, services, project state |
| `USER.md` | Human preferences: language, timezone, role |
| `load_memory_block()` | Reads both files; returns labelled sections joined by `\n\n` |
| System prompt injection | Memory appears in the system message — stable across all turns, never compressed |
| Write-back | Agent can update `MEMORY.md` via `write_file` tool to persist new facts |
| Missing files | Sections are silently omitted — no error |

**ch10** covers the **closed learning loop** — how complex sessions automatically
produce `SKILL.md` files, closing the loop between execution and future reuse.
