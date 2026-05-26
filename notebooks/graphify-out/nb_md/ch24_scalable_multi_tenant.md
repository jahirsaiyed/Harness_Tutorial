# ch24_scalable_multi_tenant

# Scalable Multi-Tenant Architecture

Harness Agent tutorial — `ch24_scalable_multi_tenant.ipynb`

## Chapter objectives

- Understand why **SQLite is the right tutorial storage** and the exact conditions that require graduating to shared storage.
- Implement a `TenantContext` that **namespaces sessions, skills, and memory per tenant** — no cross-tenant data leakage.
- Build a **token-bucket `RateLimiter`** that enforces per-tenant LLM call quotas without external dependencies.
- Design a `SessionRouter` with an **abstract storage interface** so the tutorial SQLite backend and a production Redis/Postgres backend are interchangeable.
- Understand the **stateless agent invariant**: agent processes own no state; all state lives in the shared store.

## Prerequisites

ch00–ch23 completed or package installed. No extra packages required for this chapter.

## Concept: When SQLite is not enough

The tutorial uses SQLite for sessions. SQLite is an excellent choice for development and single-process deployments because:
- Zero configuration — it's a file
- Full ACID semantics
- Fast for reads and single-writer workloads

### The SQLite scaling wall

SQLite hits limits in production harnesses in three specific ways:

| Scenario | SQLite behaviour | Production need |
|----------|-----------------|----------------|
| Two agent processes same session | Write lock contention | Shared store (Postgres, Redis) |
| 1,000 tenants, each with a DB file | 1,000 open file handles | Single shared DB with tenant partition |
| Gateway restarts lose in-flight sessions | `_sessions` dict is in-process memory | Persistent session routing store |
| One tenant hogs all LLM calls | No quotas in tutorial | Per-tenant rate limiting |

### Multi-tenancy mental model

A **tenant** is any isolated unit — a team, a workspace, a user, a project. Multi-tenancy means:

```
Tenant A:  sessions/  skills/  MEMORY.md   ← isolated
Tenant B:  sessions/  skills/  MEMORY.md   ← isolated
Tenant C:  sessions/  skills/  MEMORY.md   ← isolated
                │
                ▼
       Shared agent processes (stateless)
       Shared session store   (stateful, external)
       Shared provider registry (read-only config)
```

### Stateless agent invariant

For horizontal scaling to work, **agent processes must own no mutable state**. Every piece of mutable state — sessions, memory, skills — must live in the external store:

```
Process 1 (agent)  ──┐
Process 2 (agent)  ──┼──► Shared SessionStore (Redis / Postgres)
Process 3 (agent)  ──┘
```

Any process can handle any turn for any tenant. No sticky sessions, no local caches.

### Token-bucket rate limiting

A **token bucket** allows bursty traffic up to a capacity, then throttles:

```
Bucket capacity: 10 calls
Refill rate:     2 calls/second

t=0:  tenant sends 10 calls → all succeed (bucket: 0)
t=1:  tenant sends 3 calls  → 2 succeed (bucket refilled to 2), 1 rejected
t=6:  tenant sends 5 calls  → all succeed (bucket refilled to 10, capped)
```

Per-tenant buckets ensure one tenant can't starve others.

## How it works — annotated source

### TenantContext

```python
# multi_tenant/context.py

from dataclasses import dataclass
from pathlib import Path

@dataclass
class TenantContext:
    tenant_id: str
    base_home: Path          # e.g. labs/

    @property
    def home(self) -> Path:
        """Namespaced home directory: labs/tenants/<tenant_id>/"""
        return self.base_home / "tenants" / self.tenant_id

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def memory_path(self) -> Path:
        return self.home / "MEMORY.md"

    def ensure_dirs(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
```

### RateLimiter (token bucket)

```python
# multi_tenant/rate_limiter.py

import time
from dataclasses import dataclass, field
from threading import Lock

@dataclass
class TokenBucket:
    capacity: int              # max tokens in bucket
    refill_rate: float         # tokens added per second
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def try_acquire(self, n: int = 1) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


class RateLimiter:
    def __init__(self, capacity: int = 10, refill_rate: float = 1.0):
        self._capacity = capacity
        self._rate = refill_rate
        self._buckets: dict[str, TokenBucket] = {}     # one bucket per tenant

    def try_acquire(self, tenant_id: str) -> bool:
        if tenant_id not in self._buckets:
            self._buckets[tenant_id] = TokenBucket(self._capacity, self._rate)
        return self._buckets[tenant_id].try_acquire()
```

### SessionRouter (abstract interface)

```python
# multi_tenant/session_router.py

from abc import ABC, abstractmethod
from typing import Optional

class AbstractSessionStore(ABC):
    @abstractmethod
    def get(self, routing_key: str) -> Optional[str]: ...
    # Returns the session_id for this routing_key, or None

    @abstractmethod
    def set(self, routing_key: str, session_id: str, ttl_seconds: int = 3600) -> None: ...
    # Associates routing_key → session_id with an expiry

    @abstractmethod
    def delete(self, routing_key: str) -> None: ...


class InMemorySessionStore(AbstractSessionStore):
    """Tutorial / test implementation — not suitable for multi-process deployment."""
    def __init__(self):
        self._store: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def set(self, key: str, session_id: str, ttl_seconds: int = 3600) -> None:
        self._store[key] = session_id

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
```

## Reference implementation map

| Harness Agent | Production equivalent | Purpose |
|--------------|----------------------|--------|
| `TenantContext` | Workspace/org model in SaaS products | Namespace isolation |
| `RateLimiter` (in-process) | Redis `INCR` + `EXPIRE`, token bucket in API gateway | Per-tenant quota enforcement |
| `InMemorySessionStore` | Redis `SETEX` / `GET` | Session routing across processes |
| `AbstractSessionStore` | Repository pattern | Swap storage without changing business logic |

In production: replace `InMemorySessionStore` with a Redis-backed implementation. The `AbstractSessionStore` interface ensures `SessionRouter` never changes.

## Design choices

| Choice | Rationale |
|--------|-----------|
| `labs/tenants/<id>/` directory namespacing | Works with the existing file-based `SessionStore` and `SkillCatalog` — zero code changes to core |
| Token bucket over fixed window | Allows short bursts (better UX) while enforcing long-run average rate |
| In-process `RateLimiter` | Sufficient for single-process tutorial; swap to Redis `INCR` for multi-process |
| Abstract `SessionStore` interface | Business logic (SessionRouter) unchanged when swapping SQLite → Redis |
| TTL on session routing entries | Stale routing keys expire automatically — no explicit cleanup needed |
| Stateless agent processes | Enables horizontal scaling without sticky sessions or distributed locks |
| Per-tenant `ensure_dirs()` on first access | Lazy initialisation — no startup cost for tenants that never use the agent |

## Implementation walkthrough

```python
import os
from dataclasses import dataclass
from pathlib import Path
os.environ.setdefault('HARNESS_AGENT_HOME', 'labs')

# ── TenantContext ────────────────────────────────────────────────────────────
@dataclass
class TenantContext:
    tenant_id: str
    base_home: Path

    @property
    def home(self) -> Path:
        return self.base_home / "tenants" / self.tenant_id

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def memory_path(self) -> Path:
        return self.home / "MEMORY.md"

    def ensure_dirs(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)


# ── Demo: two tenants, fully isolated directories ────────────────────────────
base = Path("labs")
tenant_a = TenantContext("team-alpha", base)
tenant_b = TenantContext("team-beta",  base)

tenant_a.ensure_dirs()
tenant_b.ensure_dirs()

for t in [tenant_a, tenant_b]:
    print(f"Tenant: {t.tenant_id}")
    print(f"  home     : {t.home}")
    print(f"  sessions : {t.sessions_dir}  exists={t.sessions_dir.exists()}")
    print(f"  skills   : {t.skills_dir}    exists={t.skills_dir.exists()}")
    print(f"  memory   : {t.memory_path}")
    print()

# Verify isolation: writing alpha's MEMORY.md does not affect beta
tenant_a.memory_path.write_text("Team Alpha memory: we prefer Python 3.12\n")
alpha_mem = tenant_a.memory_path.read_text()
beta_mem = tenant_b.memory_path.read_text() if tenant_b.memory_path.exists() else "(empty)"
print(f"Alpha MEMORY.md: {alpha_mem.strip()}")
print(f"Beta  MEMORY.md: {beta_mem}")
```

```python
import time
from dataclasses import dataclass, field
from threading import Lock

# ── Token-bucket RateLimiter ─────────────────────────────────────────────────
@dataclass
class TokenBucket:
    capacity: int
    refill_rate: float          # tokens per second
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(float(self.capacity), self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def try_acquire(self, n: int = 1) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    @property
    def tokens(self) -> float:
        with self._lock:
            self._refill()
            return round(self._tokens, 2)


class RateLimiter:
    def __init__(self, capacity: int = 5, refill_rate: float = 1.0):
        self._capacity = capacity
        self._rate = refill_rate
        self._buckets: dict[str, TokenBucket] = {}

    def try_acquire(self, tenant_id: str) -> bool:
        if tenant_id not in self._buckets:
            self._buckets[tenant_id] = TokenBucket(self._capacity, self._rate)
        return self._buckets[tenant_id].try_acquire()

    def tokens_remaining(self, tenant_id: str) -> float:
        if tenant_id not in self._buckets:
            return float(self._capacity)
        return self._buckets[tenant_id].tokens


# ── Demo: burst then throttle ────────────────────────────────────────────────
limiter = RateLimiter(capacity=5, refill_rate=2.0)  # 5 burst, 2 per second refill

print("=== Burst phase (6 calls) ===")
for i in range(6):
    result = limiter.try_acquire("team-alpha")
    remaining = limiter.tokens_remaining("team-alpha")
    status = "ALLOWED" if result else "REJECTED"
    print(f"  Call {i+1}: {status:8}  tokens_remaining={remaining}")

print("\n=== Wait 1 second (refills 2 tokens) ===")
time.sleep(1.0)
print(f"  tokens_remaining after 1s: {limiter.tokens_remaining('team-alpha')}")

print("\n=== Tenant isolation: beta has its own bucket ===")
for i in range(3):
    result = limiter.try_acquire("team-beta")
    print(f"  Beta call {i+1}: {'ALLOWED' if result else 'REJECTED'}")
```

```python
import uuid, time
from abc import ABC, abstractmethod
from typing import Optional

# ── AbstractSessionStore and implementations ─────────────────────────────────
class AbstractSessionStore(ABC):
    @abstractmethod
    def get(self, routing_key: str) -> Optional[str]: ...

    @abstractmethod
    def set(self, routing_key: str, session_id: str, ttl_seconds: int = 3600) -> None: ...

    @abstractmethod
    def delete(self, routing_key: str) -> None: ...


class InMemorySessionStore(AbstractSessionStore):
    """Single-process implementation — used in tutorial and tests."""
    def __init__(self):
        self._store: dict[str, tuple[str, float]] = {}  # key → (session_id, expires_at)

    def get(self, key: str) -> Optional[str]:
        if key not in self._store:
            return None
        session_id, expires_at = self._store[key]
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return session_id

    def set(self, key: str, session_id: str, ttl_seconds: int = 3600) -> None:
        self._store[key] = (session_id, time.monotonic() + ttl_seconds)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


class RedisSessionStoreStub(AbstractSessionStore):
    """Shows the production interface — wire to redis-py in real deployment."""
    def __init__(self, url: str = "redis://localhost:6379"):
        self._url = url
        # In production: self._redis = redis.from_url(url)

    def get(self, key: str) -> Optional[str]:
        # return self._redis.get(key)  # returns bytes or None
        raise NotImplementedError("Wire to redis-py: pip install redis")

    def set(self, key: str, session_id: str, ttl_seconds: int = 3600) -> None:
        # self._redis.setex(key, ttl_seconds, session_id)
        raise NotImplementedError("Wire to redis-py: pip install redis")

    def delete(self, key: str) -> None:
        # self._redis.delete(key)
        raise NotImplementedError("Wire to redis-py: pip install redis")


# ── SessionRouter: uses AbstractSessionStore ─────────────────────────────────
class SessionRouter:
    """
    Routes incoming messages to the correct session.
    Replaces the in-process `_sessions` dict in GatewayRunner.
    """
    def __init__(self, store: AbstractSessionStore):
        self._store = store

    def get_or_create(self, routing_key: str, ttl_seconds: int = 3600) -> tuple[str, bool]:
        """Returns (session_id, created_new)."""
        existing = self._store.get(routing_key)
        if existing:
            return existing, False
        session_id = str(uuid.uuid4())
        self._store.set(routing_key, session_id, ttl_seconds)
        return session_id, True

    def close_session(self, routing_key: str) -> None:
        self._store.delete(routing_key)


# ── Demo ─────────────────────────────────────────────────────────────────────
store = InMemorySessionStore()
router = SessionRouter(store)

# Simulate three messages from two users
messages = [
    ("user:alice", "Hello"),
    ("user:bob",   "Hi there"),
    ("user:alice", "Follow-up question"),  # same session as first Alice message
    ("user:bob",   "Another question"),
]

print("=== Session routing demo ===")
for routing_key, text in messages:
    sid, is_new = router.get_or_create(routing_key)
    label = "NEW" if is_new else "CONTINUED"
    print(f"  {routing_key:<15} [{label}] session_id={sid[:8]}...  msg={text!r}")

print("\n=== Alice closes session ===")
router.close_session("user:alice")
sid, is_new = router.get_or_create("user:alice")
print(f"  alice next message: [{'NEW' if is_new else 'CONTINUED'}] session_id={sid[:8]}...")
```

```python
# ── Full integration: TenantContext + RateLimiter + SessionRouter ────────────
from dataclasses import dataclass
from typing import Optional


@dataclass
class TurnRequest:
    tenant_id: str
    user_id: str
    text: str


class MultiTenantGateway:
    """
    A gateway that enforces tenant isolation and per-tenant rate limits
    before dispatching to the agent.
    """
    def __init__(
        self,
        base_home: Path,
        limiter: RateLimiter,
        router: SessionRouter,
    ):
        self._base_home = base_home
        self._limiter = limiter
        self._router = router

    def handle(self, req: TurnRequest) -> dict:
        # 1. Rate check
        if not self._limiter.try_acquire(req.tenant_id):
            return {"ok": False, "error": "rate_limit_exceeded", "tenant": req.tenant_id}

        # 2. Tenant context
        ctx = TenantContext(req.tenant_id, self._base_home)
        ctx.ensure_dirs()

        # 3. Session routing
        routing_key = f"{req.tenant_id}:{req.user_id}"
        session_id, is_new = self._router.get_or_create(routing_key)

        # 4. (In real code) set HARNESS_AGENT_HOME to ctx.home and call agent
        # agent = AIAgent(session_id=session_id)
        # result = agent.run_conversation(req.text)

        return {
            "ok": True,
            "tenant": req.tenant_id,
            "session_id": session_id[:8] + "...",
            "new_session": is_new,
            "home": str(ctx.home),
            "tokens_remaining": self._limiter.tokens_remaining(req.tenant_id),
        }


# ── Demo ─────────────────────────────────────────────────────────────────────
gateway = MultiTenantGateway(
    base_home=Path("labs"),
    limiter=RateLimiter(capacity=3, refill_rate=1.0),
    router=SessionRouter(InMemorySessionStore()),
)

requests = [
    TurnRequest("acme-corp", "alice", "Summarise today's logs"),
    TurnRequest("acme-corp", "alice", "Now create a ticket"),
    TurnRequest("acme-corp", "bob",   "What's the build status?"),
    TurnRequest("acme-corp", "carol", "Help me with Python"),   # will be rate-limited
    TurnRequest("other-org", "dave",  "Different tenant — own bucket"),
]

print("=== MultiTenantGateway demo ===")
for req in requests:
    result = gateway.handle(req)
    if result["ok"]:
        print(f"  [{req.tenant_id}/{req.user_id}] OK session={result['session_id']} "
              f"new={result['new_session']} tokens={result['tokens_remaining']}")
    else:
        print(f"  [{req.tenant_id}/{req.user_id}] REJECTED: {result['error']}")
```

```python
# ── Scaling invariant demo: two "agent processes" sharing one SessionRouter ───
import uuid

# Shared state (in production: Redis)
shared_store = InMemorySessionStore()
shared_router = SessionRouter(shared_store)

# Process 1 handles Alice's first message
print("=== Process 1 handles Alice turn 1 ===")
sid_p1, new_p1 = shared_router.get_or_create("user:alice")
print(f"  Process 1: session={sid_p1[:8]}... new={new_p1}")
# Process 1 stores conversation in the session (via SessionStore — ch06)
# In this demo we just record that it happened

# Process 2 (different process/machine) handles Alice's second message
print("\n=== Process 2 handles Alice turn 2 ===")
sid_p2, new_p2 = shared_router.get_or_create("user:alice")
print(f"  Process 2: session={sid_p2[:8]}... new={new_p2}")

# Both processes resolve to the SAME session_id — multi-turn continuity preserved
assert sid_p1 == sid_p2, "Session continuity broken!"
print(f"\n  Session IDs match: {sid_p1[:8]} == {sid_p2[:8]}  ✓")
print("  Multi-turn continuity preserved across stateless processes.")

print("\n=== Migration guide: SQLite → Redis ===")
print("""
  1. pip install redis
  2. Implement RedisSessionStore(AbstractSessionStore):
       get()    → self._redis.get(key).decode()
       set()    → self._redis.setex(key, ttl, session_id)
       delete() → self._redis.delete(key)
  3. Replace InMemorySessionStore() with RedisSessionStore(url)
  4. SessionRouter and MultiTenantGateway: ZERO changes needed.
""")
```

## Trace: multi-tenant request path

```
Incoming message (tenant_id="acme", user_id="alice", text="...")
    │
    ▼
MultiTenantGateway.handle(req)
    ├─ RateLimiter.try_acquire("acme")
    │     ├─ bucket["acme"]._refill()       ← elapsed time adds tokens
    │     └─ returns True/False             ← False → 429 response
    │
    ├─ TenantContext("acme", base_home)
    │     └─ home = labs/tenants/acme/      ← isolated from all other tenants
    │
    ├─ SessionRouter.get_or_create("acme:alice")
    │     └─ AbstractSessionStore.get()     ← SQLite or Redis, same interface
    │
    └─ AIAgent(session_id, home=ctx.home).run_conversation(text)
          └─ All subsystems use ctx.home → tenant-isolated
```

## Hands-on exercises

1. **Add a `quota_per_hour` field**: Extend `TenantContext` with a `quota_per_hour: int` field. Wire it into `RateLimiter` so each tenant uses its own capacity setting.

2. **Redis backend**: Implement `RedisSessionStore(AbstractSessionStore)` using `redis-py`. Verify that `SessionRouter` requires zero changes.
   ```bash
   pip install redis
   docker run -p 6379:6379 redis:alpine
   ```

3. **TTL expiry test**: Create a session with `ttl_seconds=1`, sleep 2 seconds, then call `router.get_or_create()` again. Verify a new session is created.

4. **Tenant listing**: Add a `list_tenants(base_home)` function that returns all tenant IDs by scanning `labs/tenants/`. Use it to build a simple admin dashboard.

5. **Wire into GatewayRunner**: Modify `harness_agent/gateway/runner.py` to use `SessionRouter` backed by `InMemorySessionStore` instead of the plain `_sessions` dict. Verify `harness-agent gateway run` still works.

6. **Rate limit headers**: Modify `MultiTenantGateway.handle()` to include `X-RateLimit-Remaining` and `X-RateLimit-Reset` in the response (as it would in an HTTP response). Test with a burst of calls.

## Common pitfalls

| Pitfall | Symptom | Diagnosis |
|---------|---------|----------|
| `HARNESS_AGENT_HOME` not set per-tenant | All tenants share the same `labs/` | Pass `ctx.home` explicitly to `AIAgent`; don't rely on env var for multi-tenant |
| In-process `RateLimiter` across multiple processes | Quota not enforced — each process has its own bucket | Use Redis `INCR` + `EXPIRE` for multi-process rate limiting |
| Session routing key collision | Two users share a session | Routing key must include both tenant and user: `f"{tenant}:{user}"` |
| Missing TTL on session entries | Store grows unbounded | Always pass `ttl_seconds` to `set()` |
| Tenant directory not created before agent runs | `FileNotFoundError` on first skill save | Call `ctx.ensure_dirs()` before dispatching to agent |
| Shared `SkillCatalog` across tenants | Skills leak between tenants | Pass `skills_dir=ctx.skills_dir` to `SkillCatalog` |
| `InMemorySessionStore` in production | Sessions lost on restart | Use `RedisSessionStore` or `SqliteSessionStore` for production |

## Checkpoint questions

1. Name three scenarios where SQLite is insufficient for a production harness.
2. What is the stateless agent invariant? Why does it enable horizontal scaling?
3. In the token bucket algorithm, what is the difference between `capacity` and `refill_rate`? Which controls burst?
4. Why does `TenantContext.home` return `labs/tenants/<id>/` rather than just `labs/<id>/`?
5. What must change when you swap `InMemorySessionStore` for `RedisSessionStore`? What stays the same?
6. A tenant sends 100 requests in 1 second with a bucket of capacity=10, refill_rate=2. How many requests succeed?
7. Explain how `SessionRouter.get_or_create()` provides multi-turn continuity across stateless processes.

## Summary

| Concept | Key detail |
|---------|----------|
| `TenantContext` | Namespaces `home`, `sessions`, `skills`, `MEMORY.md` per tenant under `labs/tenants/<id>/` |
| `TokenBucket` | Token bucket: `capacity` sets burst limit, `refill_rate` sets sustained rate |
| `RateLimiter` | One bucket per tenant — isolated quota enforcement |
| `AbstractSessionStore` | Interface with `get` / `set` / `delete` — swap SQLite for Redis without touching `SessionRouter` |
| `InMemorySessionStore` | Tutorial/test implementation — supports TTL expiry |
| `SessionRouter` | Maps routing key → session_id — enables multi-turn continuity across stateless processes |
| `MultiTenantGateway` | Composes rate limiter + tenant context + session router before dispatching to agent |
| Stateless agent invariant | Agent processes own no state — all state in external store — enables horizontal scaling |

**ch25** covers reliability and resilience — circuit breakers, retry strategies, fallback provider chains, and chaos testing.
