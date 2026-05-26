# ch15_gateway_and_cli

# Gateway and CLI

Harness Agent tutorial — `ch15_gateway_and_cli.ipynb`


## Chapter objectives

- Understand how `GatewayRunner` maps HTTP webhooks to agent conversations.
- Inspect the `platform:user_id` → `session_id` routing table.
- Trace an HTTP POST through `handle_message()` to the shared `AIAgent`.
- Compare gateway (multi-user, stateful) versus CLI (single-user REPL) entry points.


## Prerequisites

Prior chapters through ch15; see SYLLABUS.md.


## Concept: Gateway and CLI

Harness Agent has multiple **entry points** that all funnel into the same `AIAgent`. The gateway and CLI are the two most user-facing ones.

### Gateway (`gateway/runner.py`)

The gateway accepts HTTP POST requests and routes them to the right conversation session based on `(platform, user_id)`:

```
POST / HTTP/1.1
Content-Type: application/json

{"text": "What is 2+2?", "user_id": "alice", "platform": "slack"}
```

**Routing table** — `_sessions: dict[str, str]`:

```
key = f"{platform}:{user_id}"   # e.g. "slack:alice"
session_id = _sessions.get(key) # None on first message
result = agent.run_conversation(text, session_id=session_id)
_sessions[key] = result.session_id  # persist for next message
```

Because `GatewayRunner` holds a single `AIAgent` instance, **all users share the same tool registry and provider** but maintain **separate SQLite sessions** (keyed by `session_id`). Memory is per-session, not global.

### HTTP server (`run_http()`)

Uses Python's stdlib `HTTPServer` — no external framework needed:

```python
server = HTTPServer((self.host, self.port), Handler)
server.serve_forever()  # blocks; run via: harness-agent gateway run
```

Defaults: `127.0.0.1:8765` (override with `HARNESS_GATEWAY_HOST` / `HARNESS_GATEWAY_PORT`).

### CLI entry point (`harness-agent chat`)

The CLI is a REPL that creates a **single session** for one user:

```
harness-agent chat
> Hello!
Agent: Hi! How can I help you today?
> /model anthropic claude-haiku-4-5-20251001
> ...
```

| Entry point | Session routing | Users | Persistent session |
|-------------|----------------|-------|--------------------|
| `gateway run` | `platform:user_id` → session_id | many | yes |
| `chat` | one session per invocation | one | yes |
| `cron tick` | isolated (no persist) | scheduler | no |
| `acp` | one session per stdio stream | IDE | yes |


## How it works — annotated source

```python
# gateway/runner.py — GatewayRunner

class GatewayRunner:
    def __init__(self, host=None, port=None):
        self.host = host or os.environ.get("HARNESS_GATEWAY_HOST", "127.0.0.1")
        self.port = port or int(os.environ.get("HARNESS_GATEWAY_PORT", "8765"))
        self.agent = AIAgent()              # (1) shared agent — tool registry loaded once
        self._sessions: dict[str, str] = {}  # (2) platform:user → session_id

    def handle_message(self, platform, user_id, text) -> str:
        key = f"{platform}:{user_id}"       # (3) composite routing key
        session_id = self._sessions.get(key)# (4) None on first message
        result = self.agent.run_conversation(text, session_id=session_id)
        if result.session_id:
            self._sessions[key] = result.session_id  # (5) persist for continuity
        return result.assistant_text

    def run_http(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                data = json.loads(body)     # (6) parse JSON body
                reply = runner.handle_message(
                    data.get("platform", "webhook"),
                    data.get("user_id", "default"),
                    data.get("text", ""),
                )
                payload = json.dumps({"reply": reply}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(payload)   # (7) return JSON reply
        HTTPServer((self.host, self.port), Handler).serve_forever()
```

```mermaid
flowchart LR
  Client -->|POST /| GW[GatewayRunner]
  GW -->|platform:user → session_id| Sessions[_sessions dict]
  GW -->|run_conversation| Agent[AIAgent]
  Agent --> SQLite[(sessions DB)]
  Agent -->|text| GW
  GW -->|{"reply": text}| Client
```


## Reference implementation map

| Harness Agent | Nous Research agent (`REFERENCE_REPO_PATH`) | OpenClaw |
|---------------|---------------------------------------------|----------|
| ``gateway/runner.py`, `cli/main.py`` | search architecture guide | SOUL/gateway patterns |

Open upstream files only under your optional clone — not bundled in this tutorial.


## Design choices

| Choice | Rationale |
|--------|-----------|
| Stdlib `HTTPServer` | No external dependency; easy to replace with FastAPI or aiohttp |
| Single shared `AIAgent` | Tool registry loaded once, not per request |
| `platform:user_id` composite key | Same agent from different platforms stays separate |
| Session state in `_sessions` dict | In-memory; restart loses routing (acceptable for tutorial) |
| `platform` defaults to `"webhook"` | Generic fallback so any POST body without platform field works |
| REPL for CLI | Simplest multi-turn interface; `/model` command switches provider live |

**Extension points:**
- Replace `HTTPServer` with an async ASGI app for concurrent requests.
- Persist `_sessions` to Redis or SQLite to survive restarts.
- Add platform-specific adapters (Slack event format, Discord webhooks, etc.).


## Implementation walkthrough


```python
import os
os.environ.setdefault('HARNESS_AGENT_HOME', 'labs')

from harness_agent.gateway.runner import GatewayRunner

# Inspect default configuration
r = GatewayRunner()
print(f"Gateway host : {r.host}")
print(f"Gateway port : {r.port}")
print(f"Sessions dict: {r._sessions}")  # empty on startup

# Demonstrate the routing key logic (no API call needed)
platform = "slack"
user_id   = "alice"
key = f"{platform}:{user_id}"
print(f"\nRouting key for ({platform!r}, {user_id!r}): {key!r}")
print("First message  → session_id = None (new session will be created)")
print("Second message → session_id = <uuid from first turn>")

```

## Trace: simulating the gateway session routing


```python
import os
os.environ.setdefault('HARNESS_AGENT_HOME', 'labs')

# Simulate the _sessions routing table without making real HTTP calls
sessions: dict[str, str] = {}

def get_or_create_session(sessions, platform, user_id, returned_session_id):
    """Mirrors GatewayRunner.handle_message() routing logic."""
    key = f"{platform}:{user_id}"
    existing = sessions.get(key)
    print(f"  key={key!r}  existing_session={existing!r}")
    # After the agent call, store the returned session_id
    sessions[key] = returned_session_id
    return existing  # None means "new session"

# Simulate 3 messages from 2 different users
print("Message 1: alice/slack (first message)")
sid = get_or_create_session(sessions, "slack", "alice", "sess-alice-001")
print(f"  session passed to agent: {sid!r}  (None → agent creates new session)")

print("\nMessage 2: bob/slack (first message from bob)")
sid = get_or_create_session(sessions, "slack", "bob", "sess-bob-001")
print(f"  session passed to agent: {sid!r}  (None → separate new session for bob)")

print("\nMessage 3: alice/slack (follow-up from alice)")
sid = get_or_create_session(sessions, "slack", "alice", "sess-alice-001")
print(f"  session passed to agent: {sid!r}  (continues alice's session)")

print("\nFinal sessions table:", sessions)
print("\nKey insight: alice and bob have separate session histories even")
print("though they share the same AIAgent instance.")

```

## Hands-on exercises

1. **Start the gateway**: Run `harness-agent gateway run` in a terminal. In another terminal, send a POST with `curl -X POST http://127.0.0.1:8765 -H "Content-Type: application/json" -d '{"text":"hello","user_id":"me","platform":"test"}'`. Observe the JSON response.

2. **Multi-turn continuity**: Send two POSTs with the same `user_id` but different `text`. Check `GatewayRunner._sessions` (or the sessions DB) to confirm both turns landed in the same session.

3. **Platform isolation**: Send identical text from two different `platform` values (`"slack"` vs `"discord"`) with the same `user_id`. Confirm they create separate sessions — the routing key `platform:user_id` keeps them apart.

4. **CLI multi-turn**: Run `harness-agent chat` and have a 3-turn conversation. Then use `search_sessions` tool (ch06) to confirm all three turns are in one session.

5. **Custom port**: Set `HARNESS_GATEWAY_PORT=9000` and restart the gateway. Verify `GatewayRunner().port == 9000` and that `curl` hits the new port.

6. **Async upgrade**: Replace `HTTPServer` with `FastAPI` + `uvicorn` to handle concurrent requests. The `handle_message()` logic stays unchanged — only the HTTP layer changes.


## Common pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Missing `Content-Length` header | Body reads as empty | `curl` sets it automatically; custom clients must include it |
| Sessions lost on restart | Every message starts a new session | `_sessions` is in-memory; persist to Redis/DB for production |
| Port already in use | `OSError: [Errno 98] Address in use` | Change `HARNESS_GATEWAY_PORT` or kill the process holding the port |
| Concurrent requests blocking | Second request waits for first | `HTTPServer` is single-threaded; use `ThreadingHTTPServer` or async |
| Wrong `platform` in routing key | Two clients share a session | Ensure `platform` in POST body is consistent across requests |
| JSON decode failure | 400 / empty reply | Gateway falls back to treating body as plain text — send valid JSON |
| CLI `/model` typo | `KeyError: Unknown provider` | List providers: `list(get_provider_registry()._providers.keys())` |


## Checkpoint questions

1. What is the composite routing key used by `GatewayRunner`? Give an example.
2. If Alice sends her first message, what is `session_id` passed to `run_conversation()`? What about her second message?
3. How many `AIAgent` instances does one `GatewayRunner` hold? Why?
4. What Python stdlib class serves HTTP requests in `run_http()`? What is its main limitation for production use?
5. What are the two env vars that control gateway host and port? What are their defaults?
6. How does the CLI differ from the gateway in terms of session routing and user count?
7. If the gateway restarts, what session information is lost and what is preserved?


## Summary

| Concept | Key detail |
|---------|-----------|
| `GatewayRunner` | Shared `AIAgent` + in-memory `_sessions: dict[str, str]` |
| Routing key | `f"{platform}:{user_id}"` — composite key keeps platforms separate |
| Session continuity | `session_id` returned by first turn is stored and reused |
| HTTP server | stdlib `HTTPServer` on `127.0.0.1:8765` (no external dependencies) |
| POST body | `{text, user_id, platform}` → response: `{reply}` |
| Config env vars | `HARNESS_GATEWAY_HOST`, `HARNESS_GATEWAY_PORT` |
| CLI entry | `harness-agent chat` — single-user REPL with `/model` command |
| ACP entry | `harness-agent acp` — stdio JSON-RPC for IDE integration (ch18) |

**ch16** dives into provider resolution — how the agent decides which LLM API to call and how to find the API key.

