# ch07_prompt_assembly

# Prompt assembly

Harness Agent tutorial — `ch07_prompt_assembly.ipynb`


## Chapter objectives

By the end of this chapter you will be able to:

- List the five sections of the system prompt in assembly order.
- Trace `PromptBuilder.build_system_prompt()` from source to output string.
- Explain why the system prompt is built **once** before the agent loop (prompt caching).
- Describe what each file contributes: `SOUL.md`, `MEMORY.md`, `USER.md`, `SKILL.md`, `.harness.md`.
- Predict which sections are present with a fresh `HARNESS_AGENT_HOME` and which require files to exist.
- Inject a `workspace_context` string and verify it appears in the assembled prompt.

## Prerequisites

Prior chapters through ch07; see SYLLABUS.md.


## Concept: Prompt assembly

### Why a dedicated builder?

The system prompt is the agent's "personality + knowledge" at the start of every
conversation. It must be:

- **Stable**: never mutated mid-conversation (breaks caching and reproducibility).
- **Composable**: different files contribute different sections.
- **Progressive**: skills are listed as metadata, not full bodies, to save tokens.

`PromptBuilder` handles all of this in one call.

### Assembly order (left to right = top to bottom in prompt)

```text
1. Base instructions   "You are Harness Agent…"  (always present)
2. SOUL.md             Persona — agent character, communication style
3. MEMORY.md           Long-term environment facts (installed tools, project state)
4. USER.md             User preferences (timezone, preferred language, etc.)
5. Skill metadata      One-line descriptions of available SKILL.md files
6. .harness.md         Project-specific context from workspace root
7. workspace_context   Optional runtime string (passed by caller)
```

Each section is only included if the corresponding file exists (`_read_optional`
returns `""` for missing files). Sections are joined with `"\n\n"`.

### Why build once before the loop?

```python
# agent.py — OUTSIDE the for loop
system = self.prompt_builder.build_system_prompt()
messages = [Message(role="system", content=system)]
```

Building inside the loop would:
1. Re-read files on every turn (I/O cost).
2. Change the system message content mid-conversation, which invalidates
   prefix-cached tokens on APIs that support prompt caching.

### Skill progressive disclosure

`SkillCatalog.metadata_block()` produces one line per skill:

```
# Available skills (metadata only — request full skill when needed)
- **my-workflow**: Steps for deploying the backend service
```

The model sees skill *names* and *descriptions* but not their full bodies.
When the model needs the body, it calls the `load_skill` tool (ch08).

## How it works

`PromptBuilder.build_system_prompt()` fixed order; no mid-turn mutation.

```mermaid
flowchart LR
  U[User or scheduler] --> A[AIAgent]
  A --> M[Prompt assembly]
```

Trace cells below execute real code paths offline where possible.


## Reference implementation map

| Harness Agent | Nous Research agent (`REFERENCE_REPO_PATH`) | OpenClaw |
|---------------|---------------------------------------------|----------|
| ``prompt/builder.py`` | search architecture guide | SOUL/gateway patterns |

Open upstream files only under your optional clone — not bundled in this tutorial.


## Design choices in harness_agent

Tutorial implementation prioritizes readable Python over feature parity. Extend ``prompt/builder.py`` as exercises.


## Implementation walkthrough


```python
from harness_agent.prompt.builder import PromptBuilder
from harness_agent.config import get_config
from pathlib import Path

config = get_config()
builder = PromptBuilder()

# --- Build with no optional files ---
prompt = builder.build_system_prompt()
sections = prompt.split("\n\n")

print(f"=== System prompt (no optional files) ===")
print(f"Total length  : {len(prompt)} chars")
print(f"Sections (\\n\\n-separated): {len(sections)}\n")
for i, s in enumerate(sections):
    print(f"  [{i}] {s[:80]!r}{'...' if len(s) > 80 else ''}")

print()

# --- Build with workspace_context ---
prompt2 = builder.build_system_prompt(workspace_context="Project: Harness Tutorial (Python 3.12)")
print(f"=== With workspace_context ===")
print(f"Total length: {len(prompt2)} chars")
print(f"Last section: {prompt2.split(chr(10)*2)[-1]!r}")
```

## Trace one request


```python
from pathlib import Path

config = get_config()

# Write a SOUL.md and see how the prompt changes
soul_path = config.soul_path
soul_path.write_text("You speak concisely and prefer bullet points.", encoding="utf-8")

# Write a MEMORY.md
config.memory_path.write_text("## Environment\n- Python 3.12\n- OS: Linux", encoding="utf-8")

# Rebuild prompt (PromptBuilder reads files fresh each call)
prompt3 = PromptBuilder().build_system_prompt()
sections3 = prompt3.split("\n\n")

print(f"=== Prompt with SOUL.md + MEMORY.md ===")
print(f"Total length  : {len(prompt3)} chars")
print(f"Sections: {len(sections3)}\n")
for i, s in enumerate(sections3):
    print(f"  [{i}] {s[:100]!r}")

# Clean up demo files
soul_path.unlink()
config.memory_path.unlink()
print("\n(Demo files cleaned up)")
```

## Hands-on exercises

**Exercise 1 — Custom SOUL**

Write a `SOUL.md` that gives the agent a different persona (e.g. "You are a terse
senior DevOps engineer. Skip pleasantries."). Rebuild the prompt and verify the SOUL
section appears after the base instructions.

**Exercise 2 — Project context**

Create `<HARNESS_AGENT_HOME>/workspace/.harness.md` with:

```markdown
# Project: My API Server
- Language: Python 3.12
- Framework: FastAPI
- Main entry point: src/main.py
```

Rebuild the prompt and verify the project context section appears.

**Exercise 3 — Token count estimate**

Using the `estimate_tokens()` function from ch11:

```python
from harness_agent.compression.summarize import estimate_tokens
from harness_agent.types import Message
system_msg = Message(role="system", content=prompt)
print(f"System prompt: ~{estimate_tokens([system_msg])} tokens")
```

How many tokens does the base system prompt use? What happens after adding SOUL + MEMORY?

**Exercise 4 — Section order matters**

Move the skill metadata to come BEFORE the MEMORY block by editing `PromptBuilder.build_system_prompt()`.
What might break if the ordering is wrong?

## Common pitfalls

| Pitfall | Root cause | Fix |
|---------|-----------|-----|
| System prompt changes each turn | Building inside the loop | Build once before `for _ in range(max_turns)` |
| SOUL not appearing | File not at `config.soul_path` | Check `config.soul_path` — must be under `HARNESS_AGENT_HOME` |
| Skill metadata missing | No `SKILL.md` files in `skills_dir` | Run a multi-tool session to trigger learning loop (ch10) |
| `.harness.md` not loaded | File at wrong path | Must be at `<HARNESS_AGENT_HOME>/workspace/.harness.md` |
| Prompt too long | Large MEMORY.md + many skills | Keep MEMORY.md focused; use progressive disclosure for skills |
| `workspace_context` not appearing | Passed as keyword to wrong call | Use `builder.build_system_prompt(workspace_context=...)` |

## Checkpoint questions

1. **Assembly order** — List the five optional file sections in the exact order `build_system_prompt()` adds them. What joins the sections?

2. **Caching** — Why is the system prompt built before the `for` loop in `agent.py`? What API feature does this support?

3. **Missing files** — What does `_read_optional()` return when a file doesn't exist? How does this affect the assembled prompt?

4. **Skill metadata** — The skills block says "metadata only". What does the model need to do to get the full skill body? Which tool does it call?

5. **workspace_context** — Where does this parameter appear in the assembled prompt (which section, which position)?

6. **Token budget** — If MEMORY.md and USER.md each contain 2 000 chars, approximately how many tokens does the system prompt consume (using the `estimate_tokens` formula)?

## Summary & next chapter

| Topic | Key takeaway |
|-------|-------------|
| Assembly order | base → SOUL → MEMORY → USER → skills metadata → `.harness.md` → workspace_context |
| `_read_optional()` | Returns `""` for missing files — sections are conditional |
| Build once | System prompt is stable across all loop turns to support prompt caching |
| Progressive disclosure | Only skill names/descriptions in the prompt; full body on `load_skill` tool call |
| `workspace_context` | Optional runtime override injected as the last section |

**ch08** covers the **skills system** — how `SKILL.md` files are discovered, indexed
as metadata, and loaded on demand via progressive disclosure.
