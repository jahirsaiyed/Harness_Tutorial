# ch06_session_storage

# Session storage

Harness Agent tutorial — `ch06_session_storage.ipynb`


## Chapter objectives

By the end of this chapter you will be able to:

- Describe the three-table SQLite schema (`sessions`, `turns`, `turns_fts`).
- Call `create_session()`, `append_turn()`, and `load_messages()` correctly.
- Explain why FTS5 is used for session search and how `MATCH` queries work.
- Trace the data flow from `Message` → JSON payload → `turns` row → `turns_fts` index.
- Understand how `GatewayRunner` maps `(platform, user_id)` to a `session_id`.
- Inspect a real SQLite database to verify stored message payloads.

## Prerequisites

Prior chapters through ch06; see SYLLABUS.md.


## Concept: Session storage

### Why persist sessions?

Without persistence, every `run_conversation()` call starts fresh — no memory of
prior turns. `SessionStore` writes every message to SQLite so:

1. Multi-turn conversations survive across calls.
2. A gateway can route the same user to the same session.
3. Sessions can be searched for relevant context (e.g. "find sessions where I discussed auth").
4. Trajectories can be exported for fine-tuning (ch19).

### The three-table schema

```sql
sessions (id TEXT PK, title TEXT, parent_id TEXT, created_at TEXT)
turns    (id INT PK AUTOINCREMENT, session_id TEXT FK, role TEXT, content TEXT, payload TEXT)
turns_fts  -- FTS5 virtual table, indexes session_id + content
```

`turns.payload` stores the full `message.to_openai()` JSON — including `tool_calls`
and `tool_call_id`. `content` is stored separately for FTS5 indexing.

### FTS5 full-text search

`turns_fts USING fts5(session_id, content)` enables prefix and phrase search across
all stored message text:

```sql
SELECT session_id, content FROM turns_fts WHERE turns_fts MATCH 'harness agent';
```

This powers the `search_sessions` tool the model can call to retrieve relevant history.

### Session lifecycle in agent.py

```python
session_id = self.sessions.create_session(title=user_input[:60])  # once
history = self.sessions.load_messages(session_id)                  # each call
# … loop …
self.sessions.append_turn(session_id, asst_message)                # after each turn
self.sessions.append_turn(session_id, tool_message)
```

For `isolated=True` agents, none of these calls happen — history is always empty.

## How it works

Append turns after each message; search indexes user/assistant text.

```mermaid
flowchart LR
  U[User or scheduler] --> A[AIAgent]
  A --> M[Session storage]
```

Trace cells below execute real code paths offline where possible.


## Reference implementation map

| Harness Agent | Nous Research agent (`REFERENCE_REPO_PATH`) | OpenClaw |
|---------------|---------------------------------------------|----------|
| ``sessions/store.py`` | search architecture guide | SOUL/gateway patterns |

Open upstream files only under your optional clone — not bundled in this tutorial.


## Design choices in harness_agent

Tutorial implementation prioritizes readable Python over feature parity. Extend ``sessions/store.py`` as exercises.


## Implementation walkthrough


```python
from harness_agent.sessions.store import SessionStore
from harness_agent.types import Message
import json

s = SessionStore()

# Create a session
sid = s.create_session(title="ch06 demo session")
print(f"Session ID: {sid}")
print(f"DB path   : {s.db_path}\n")

# Append several turns
turns = [
    Message(role="user",      content="What is Harness Agent?"),
    Message(role="assistant", content="Harness Agent is a tutorial AI agent harness."),
    Message(role="user",      content="How does session storage work?"),
    Message(role="assistant", content="It uses SQLite with FTS5 for full-text search."),
]
for t in turns:
    s.append_turn(sid, t)

print(f"Appended {len(turns)} turns")

# Load messages back
loaded = s.load_messages(sid)
print(f"Loaded {len(loaded)} messages\n")
for m in loaded:
    print(f"  [{m.role:9s}] {m.content[:60]!r}")
```

## Trace one request


```python
# FTS5 search across sessions
hits = s.search("SQLite FTS5")
print(f"Search 'SQLite FTS5' → {len(hits)} hits")
for h in hits:
    print(f"  session={h['session_id'][:8]}…  snippet={h['snippet'][:60]!r}")

print()

# Search for another term
hits2 = s.search("tutorial")
print(f"Search 'tutorial' → {len(hits2)} hits")
for h in hits2:
    print(f"  session={h['session_id'][:8]}…  snippet={h['snippet'][:60]!r}")

print()
# Show raw payload for one turn
import sqlite3
with sqlite3.connect(s.db_path) as conn:
    row = conn.execute("SELECT role, payload FROM turns WHERE session_id = ? LIMIT 1", (sid,)).fetchone()
    if row:
        print(f"Raw payload for first turn (role={row[0]!r}):")
        print(json.dumps(json.loads(row[1]), indent=2))
```

## Hands-on exercises

**Exercise 1 — Tool messages in sessions**

Append a tool message with `tool_call_id` and verify it round-trips through
`load_messages()`:

```python
tool_msg = Message(
    role="tool",
    content='{"status": "success", "summary": "found 3 files"}',
    tool_call_id="call_abc",
    name="list_files",
)
s.append_turn(sid, tool_msg)
loaded = s.load_messages(sid)
last = loaded[-1]
print(last.role, last.tool_call_id, last.name)
```

**Exercise 2 — Parent session**

Create a child session linked to the parent:

```python
child_sid = s.create_session(title="child task", parent_id=sid)
```

Open the DB in sqlite3 and verify the `parent_id` column is set.

**Exercise 3 — search_sessions tool**

The `search_sessions` tool wraps `SessionStore.search()`. Call it directly:

```python
from harness_agent.sessions.store import search_sessions
print(search_sessions("SQLite"))
```

What observation format does it return?

**Exercise 4 — Multi-session search**

Create a second session with different content, then search for a term that spans both.
Verify multiple `session_id` values appear in the hits.

## Common pitfalls

| Pitfall | Root cause | Fix |
|---------|-----------|-----|
| History missing on second call | `isolated=True` skips `load_messages` | Use `isolated=False` for persistent agents |
| `load_messages` returns empty | Wrong `session_id` or DB not initialised | Check `SessionStore().db_path` exists |
| FTS5 search returns no results | `turns_fts` not populated | `append_turn` indexes simultaneously — don't write to `turns` directly |
| Tool messages lost | Only persisting assistant messages | `agent.py` appends both assistant and tool messages |
| DB file in wrong location | `HARNESS_AGENT_HOME` not set | Set env var or check `config.db_path` |
| `parent_id` ignored | It's metadata only — `load_messages` doesn't chain parent sessions | Chain manually if needed |

## Checkpoint questions

1. **Schema** — List the three tables in the SQLite database. What does each store? Why is `turns.payload` different from `turns.content`?

2. **FTS5** — What SQL keyword does `SessionStore.search()` use? Why is FTS5 used instead of a simple `LIKE '%query%'`?

3. **Round-trip** — A `Message` with `tool_calls` is appended and then loaded back. Which field of `Message` is used to serialise it? Which method produces the JSON?

4. **Gateway routing** — `GatewayRunner` stores `{f"{platform}:{user_id}": session_id}`. What happens when the same user sends a second message? What happens when a new user sends a message?

5. **isolated=True** — List every `SessionStore` method call that `isolated=True` suppresses in `run_conversation()`.

6. **Export** — Which `turns` rows does `export_trajectories()` skip and why? (Hint: see ch19)

## Summary & next chapter

| Topic | Key takeaway |
|-------|-------------|
| `sessions` table | One row per conversation: `id`, `title`, `parent_id` |
| `turns` table | One row per message: `role`, `content` (FTS), `payload` (full JSON) |
| `turns_fts` | FTS5 virtual table enabling full-text `MATCH` queries |
| `append_turn()` | Writes to both `turns` and `turns_fts` atomically |
| `load_messages()` | Deserialises `payload` JSON back to `Message` objects (preserves `tool_calls`) |
| `search_sessions` tool | Wraps `SessionStore.search()` as a callable tool for the model |

**ch07** covers **prompt assembly** — how `PromptBuilder` stitches together SOUL,
MEMORY, USER model, skill metadata, and project context into the system prompt.
