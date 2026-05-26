# ch25_reliability_and_resilience

# Reliability and Resilience

Harness Agent tutorial — `ch25_reliability_and_resilience.ipynb`

## Chapter objectives

- Implement a **`CircuitBreaker`** for LLM provider calls with `CLOSED → OPEN → HALF_OPEN` state transitions — stops cascading failures from a degraded provider.
- Build **`retry_with_backoff()`** using exponential backoff and jitter — handles transient failures without thundering-herd effects.
- Design a **`ProviderFallbackChain`** that automatically switches to the next provider when one trips its circuit breaker.
- Implement an **idempotent dead-letter queue (DLQ)** for failed cron jobs — failed jobs are retried without duplication.
- Write a **chaos harness** that injects failures at a configured rate and verifies system behaviour under fault conditions.

## Prerequisites

ch00–ch24 completed or package installed. No extra packages required for this chapter.

## Concept: Failures are not exceptional

In a production harness, LLM providers are external dependencies. They will:
- Return 429 (rate limit) during traffic spikes
- Return 500/503 during provider outages
- Hang without responding for 30+ seconds
- Return malformed responses occasionally

Designing for resilience means these events are **handled gracefully**, not propagated as unhandled exceptions.

### Circuit breaker pattern

A circuit breaker wraps a remote call and tracks failure counts. When failures exceed a threshold, it **opens** — subsequent calls fail immediately without hitting the remote service:

```
CLOSED (normal):
  call succeeds → stay CLOSED
  call fails    → increment failure_count
  failure_count >= threshold → → OPEN

OPEN (tripped):
  all calls fail immediately (no network call)
  after reset_timeout → → HALF_OPEN

HALF_OPEN (probing):
  one call allowed through as probe
  probe succeeds → → CLOSED (reset)
  probe fails    → → OPEN (re-trip)
```

**Why this matters for agents**: Without a circuit breaker, a degraded LLM provider causes every agent turn to hang for 30 seconds then fail. With a circuit breaker, the harness detects the degradation in 3 failures and stops calling the provider — it can fall back immediately.

### Retry with exponential backoff + jitter

Retrying immediately after failure often hits the same error. Exponential backoff waits longer between each retry:

```
Attempt 1: fails → wait 1s
Attempt 2: fails → wait 2s
Attempt 3: fails → wait 4s
Attempt 4: fails → give up
```

**Jitter** adds random noise to the wait time, preventing many clients from retrying simultaneously (**thundering herd**):

```python
wait = base_delay * (2 ** attempt) + random.uniform(0, 1)  # jitter
```

### Provider fallback chain

A harness with access to multiple providers (Anthropic, OpenAI, local Ollama) can try them in order:

```
Try anthropic/claude-haiku-4-5  → CircuitBreaker: OPEN  → skip
Try openai/gpt-4o-mini          → CircuitBreaker: CLOSED → call → success
```

The fallback chain provides **availability** even when the preferred provider is down.

### Dead-letter queue (DLQ)

Cron jobs that fail should not be silently discarded. A DLQ holds failed jobs for inspection and replay:

```
Cron job runs → fails → appended to DLQ with error + timestamp
Operator inspects DLQ → fixes root cause → replays job
```

**Idempotency**: replaying a DLQ job must not double-execute side effects. Each job needs a `job_id` that prevents duplicate processing.

## How it works — annotated source

### CircuitBreaker

```python
# resilience/circuit_breaker.py

import time
from enum import Enum, auto
from dataclasses import dataclass, field
from threading import Lock

class State(Enum):
    CLOSED    = auto()    # normal — calls go through
    OPEN      = auto()    # tripped — calls fail immediately
    HALF_OPEN = auto()    # probing — one call allowed

@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int   = 3       # failures before opening
    reset_timeout: float     = 30.0    # seconds before HALF_OPEN probe
    _state: State            = field(default=State.CLOSED, init=False)
    _failure_count: int      = field(default=0, init=False)
    _last_failure_time: float= field(default=0.0, init=False)
    _lock: Lock              = field(default_factory=Lock, init=False, repr=False)

    def call(self, fn, *args, **kwargs):
        with self._lock:
            if self._state == State.OPEN:
                if time.monotonic() - self._last_failure_time >= self.reset_timeout:
                    self._state = State.HALF_OPEN  # allow probe
                else:
                    raise CircuitOpenError(f"{self.name} circuit is OPEN")

        try:
            result = fn(*args, **kwargs)
            with self._lock:
                self._on_success()
            return result
        except Exception as exc:
            with self._lock:
                self._on_failure()
            raise

    def _on_success(self):
        self._failure_count = 0
        self._state = State.CLOSED

    def _on_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = State.OPEN

class CircuitOpenError(Exception): pass
```

### retry_with_backoff

```python
# resilience/retry.py

import time, random
from functools import wraps

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable: tuple = (Exception,),
):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable as exc:
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.1)  # 10% jitter
                    time.sleep(delay + jitter)
        return wrapper
    return decorator
```

## Reference implementation map

| Harness Agent | Industry pattern | Purpose |
|--------------|-----------------|--------|
| `CircuitBreaker` | Netflix Hystrix, resilience4j, `pybreaker` | Prevent cascading failures from degraded dependencies |
| `retry_with_backoff` | `tenacity`, AWS SDK retry config | Transient fault handling with rate-safe backoff |
| `ProviderFallbackChain` | LiteLLM fallbacks, LangChain fallback chains | High availability across multiple LLM providers |
| `DeadLetterQueue` | AWS SQS DLQ, RabbitMQ DLX | Failed job preservation and replay |
| `FlakyProvider` (chaos) | Chaos Monkey, `pytest-randomly` | Fault injection for resilience testing |

In production: use `tenacity` for retry logic, `pybreaker` for circuit breakers, and SQS DLQ or Redis lists for the dead-letter queue.

## Design choices

| Choice | Rationale |
|--------|-----------|
| `HALF_OPEN` state | Allows automatic recovery — circuit closes itself after `reset_timeout` without operator intervention |
| Per-call lock in CircuitBreaker | Thread-safe state transitions without race conditions on `_failure_count` |
| Jitter in backoff | Prevents thundering herd — many clients retrying simultaneously amplifies load on a recovering service |
| `retryable` tuple parameter | Only retry known-transient errors (RateLimitError, 503); never retry auth errors or malformed input |
| ProviderFallbackChain tries in order | First provider is preferred; fallback only on `CircuitOpenError` — not on all exceptions |
| DLQ with `job_id` | Idempotent replay — processing the same `job_id` twice is safe |
| `FlakyProvider` for chaos testing | Deterministic failure injection — verify system behaviour without needing a real provider outage |

## Implementation walkthrough

```python
import os, time
from enum import Enum, auto
from dataclasses import dataclass, field
from threading import Lock
os.environ.setdefault('HARNESS_AGENT_HOME', 'labs')

# ── CircuitBreaker ───────────────────────────────────────────────────────────
class State(Enum):
    CLOSED    = auto()
    OPEN      = auto()
    HALF_OPEN = auto()


class CircuitOpenError(Exception):
    pass


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int   = 3
    reset_timeout: float     = 30.0
    _state: State            = field(default=State.CLOSED, init=False)
    _failure_count: int      = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: Lock              = field(default_factory=Lock, init=False, repr=False)

    @property
    def state(self) -> State:
        return self._state

    def call(self, fn, *args, **kwargs):
        with self._lock:
            if self._state == State.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.reset_timeout:
                    self._state = State.HALF_OPEN
                else:
                    raise CircuitOpenError(
                        f"{self.name} circuit OPEN — will probe in {self.reset_timeout - elapsed:.1f}s"
                    )
        try:
            result = fn(*args, **kwargs)
            with self._lock:
                self._failure_count = 0
                self._state = State.CLOSED
            return result
        except CircuitOpenError:
            raise
        except Exception:
            with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.monotonic()
                if self._failure_count >= self.failure_threshold:
                    self._state = State.OPEN
            raise


# ── Demo: trip and probe ─────────────────────────────────────────────────────
cb = CircuitBreaker("anthropic", failure_threshold=3, reset_timeout=0.1)  # 0.1s for demo

def failing_call():
    raise ConnectionError("Provider timeout")

def succeeding_call():
    return "assistant response"

print("=== Tripping the circuit ===")
for i in range(5):
    try:
        cb.call(failing_call)
    except (ConnectionError, CircuitOpenError) as e:
        print(f"  Attempt {i+1}: {type(e).__name__} — circuit state: {cb.state.name}")

print("\n=== Wait for reset_timeout ===")
time.sleep(0.15)
print(f"  Circuit state: {cb.state.name} (still OPEN until probe)")

print("\n=== Probe with succeeding call (HALF_OPEN → CLOSED) ===")
try:
    result = cb.call(succeeding_call)
    print(f"  Probe result: {result!r}")
    print(f"  Circuit state: {cb.state.name}")
except Exception as e:
    print(f"  Probe failed: {e}")
```

```python
import time, random
from functools import wraps

# ── retry_with_backoff ───────────────────────────────────────────────────────
def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 0.01,  # short for demo; use 1.0 in production
    max_delay: float = 60.0,
    jitter_fraction: float = 0.1,
    retryable: tuple = (Exception,),
):
    """Decorator: retry `fn` on `retryable` exceptions with exponential backoff + jitter."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * jitter_fraction)
                    print(f"    [retry] attempt {attempt+1} failed ({type(exc).__name__}), "
                          f"retrying in {delay + jitter:.3f}s")
                    time.sleep(delay + jitter)
            raise last_exc  # unreachable but satisfies type checkers
        return wrapper
    return decorator


# ── Demo: transient failures then success ────────────────────────────────────
attempt_counter = [0]

@retry_with_backoff(max_retries=3, base_delay=0.01, retryable=(ConnectionError,))
def flaky_provider_call():
    attempt_counter[0] += 1
    if attempt_counter[0] < 3:
        raise ConnectionError("transient error")
    return f"success on attempt {attempt_counter[0]}"

print("=== retry_with_backoff demo ===")
result = flaky_provider_call()
print(f"  Final result: {result!r}")

# Demo: non-retryable error is NOT retried
print("\n=== Non-retryable error (ValueError) ===")
bad_counter = [0]

@retry_with_backoff(max_retries=3, base_delay=0.01, retryable=(ConnectionError,))
def auth_error_call():
    bad_counter[0] += 1
    raise ValueError("Invalid API key — do not retry")

try:
    auth_error_call()
except ValueError as e:
    print(f"  Raised after {bad_counter[0]} attempt(s): {e}")
    print(f"  (ValueError is not in retryable=(ConnectionError,) — correct behaviour)")
```

```python
from typing import Protocol, Any
from dataclasses import dataclass

# ── ProviderFallbackChain ────────────────────────────────────────────────────
class ProviderError(Exception):
    pass


@dataclass
class ProviderEntry:
    name: str
    model: str
    circuit_breaker: CircuitBreaker

    def complete(self, messages: list) -> str:
        """Call the real provider here — stub for demo."""
        raise NotImplementedError(f"{self.name} not implemented in demo")


class ProviderFallbackChain:
    """
    Tries providers in order. Skips providers whose circuit breaker is OPEN.
    Falls back automatically — callers see a successful response or AllProvidersExhaustedError.
    """
    def __init__(self, providers: list[ProviderEntry]):
        self._providers = providers

    def complete(self, messages: list) -> tuple[str, str]:
        """Returns (response_text, provider_name_used)."""
        last_exc = None
        for entry in self._providers:
            try:
                response = entry.circuit_breaker.call(entry.complete, messages)
                return response, entry.name
            except CircuitOpenError as e:
                print(f"    [fallback] {entry.name} circuit OPEN — skipping")
                last_exc = e
            except ProviderError as e:
                print(f"    [fallback] {entry.name} failed ({e}) — trying next")
                last_exc = e
        raise RuntimeError("All providers exhausted") from last_exc


# ── FlakyProvider: chaos injection ──────────────────────────────────────────
@dataclass
class FlakyProvider(ProviderEntry):
    failure_rate: float = 0.0      # 0.0 = never fails, 1.0 = always fails
    _call_count: int = field(default=0, init=False)

    def complete(self, messages: list) -> str:
        self._call_count += 1
        if random.random() < self.failure_rate:
            raise ProviderError(f"{self.name} injected failure #{self._call_count}")
        return f"Response from {self.name} (call #{self._call_count})"


# ── Demo: fallback chain with one tripped circuit ────────────────────────────
# Provider A: always fails (simulates a downed provider)
provider_a = FlakyProvider(
    name="anthropic",
    model="claude-haiku-4-5",
    circuit_breaker=CircuitBreaker("anthropic", failure_threshold=2, reset_timeout=60),
    failure_rate=1.0,
)
# Provider B: reliable fallback
provider_b = FlakyProvider(
    name="openai",
    model="gpt-4o-mini",
    circuit_breaker=CircuitBreaker("openai", failure_threshold=3, reset_timeout=60),
    failure_rate=0.0,
)

chain = ProviderFallbackChain([provider_a, provider_b])

print("=== ProviderFallbackChain demo ===")
messages = [{"role": "user", "content": "Hello"}]

for turn in range(5):
    try:
        response, used = chain.complete(messages)
        print(f"  Turn {turn+1}: provider={used}  response={response!r}")
    except RuntimeError as e:
        print(f"  Turn {turn+1}: ALL PROVIDERS EXHAUSTED — {e}")

print(f"\n  anthropic circuit state: {provider_a.circuit_breaker.state.name}")
print(f"  openai circuit state:    {provider_b.circuit_breaker.state.name}")
```

```python
import sqlite3, json, uuid, time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
os.environ.setdefault('HARNESS_AGENT_HOME', 'labs')

# ── DeadLetterQueue ──────────────────────────────────────────────────────────
@dataclass
class FailedJob:
    job_id: str
    payload: dict
    error: str
    failed_at: float
    attempts: int


class DeadLetterQueue:
    """
    SQLite-backed DLQ for failed cron jobs.
    Idempotent: same job_id can be written multiple times — only the latest is kept.
    """
    def __init__(self, db_path: Optional[Path] = None):
        home = Path(os.environ.get("HARNESS_AGENT_HOME", "labs"))
        self.db_path = db_path or (home / "cron" / "dlq.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dlq (
                    job_id    TEXT PRIMARY KEY,
                    payload   TEXT NOT NULL,
                    error     TEXT NOT NULL,
                    failed_at REAL NOT NULL,
                    attempts  INTEGER NOT NULL DEFAULT 1
                )
            """)

    def push(self, job_id: str, payload: dict, error: str) -> None:
        """Insert or update a failed job. Idempotent on job_id."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO dlq (job_id, payload, error, failed_at, attempts)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(job_id) DO UPDATE SET
                    error     = excluded.error,
                    failed_at = excluded.failed_at,
                    attempts  = dlq.attempts + 1
            """, (job_id, json.dumps(payload), error, time.time()))

    def list_failed(self) -> list[FailedJob]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dlq ORDER BY failed_at DESC"
            ).fetchall()
        return [
            FailedJob(
                job_id=r["job_id"],
                payload=json.loads(r["payload"]),
                error=r["error"],
                failed_at=r["failed_at"],
                attempts=r["attempts"],
            )
            for r in rows
        ]

    def replay(self, job_id: str) -> Optional[FailedJob]:
        """Remove from DLQ and return the job for replay. None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dlq WHERE job_id = ?", (job_id,)
            ).fetchone()
            if not row:
                return None
            conn.execute("DELETE FROM dlq WHERE job_id = ?", (job_id,))
        return FailedJob(
            job_id=row["job_id"],
            payload=json.loads(row["payload"]),
            error=row["error"],
            failed_at=row["failed_at"],
            attempts=row["attempts"],
        )

    def clear(self) -> int:
        with self._connect() as conn:
            n = conn.execute("DELETE FROM dlq").rowcount
        return n


# ── Demo ─────────────────────────────────────────────────────────────────────
dlq = DeadLetterQueue()
dlq.clear()  # start fresh for demo

# Simulate three failed cron jobs
jobs = [
    ("job-daily-summary",  {"task": "daily_summary",  "user": "alice"}, "ConnectionError: provider timeout"),
    ("job-weekly-report",  {"task": "weekly_report",   "user": "bob"},  "ValueError: missing config key"),
    ("job-daily-summary",  {"task": "daily_summary",  "user": "alice"}, "ConnectionError: provider timeout"),  # retry of same job
]

for job_id, payload, error in jobs:
    dlq.push(job_id, payload, error)
    print(f"  pushed: {job_id}")

print("\n=== DLQ contents ===")
for job in dlq.list_failed():
    print(f"  {job.job_id:<25} attempts={job.attempts}  error={job.error[:50]}")

print("\n=== Replay job-daily-summary ===")
replayed = dlq.replay("job-daily-summary")
if replayed:
    print(f"  Replaying: {replayed.job_id} payload={replayed.payload} attempts={replayed.attempts}")
    # In production: dispatch replayed.payload to the cron executor

print("\n=== DLQ after replay ===")
remaining = dlq.list_failed()
print(f"  Jobs remaining: {len(remaining)}")
for job in remaining:
    print(f"  {job.job_id}")
```

```python
# ── Chaos harness: end-to-end resilience test ─────────────────────────────────
import random

def run_chaos_test(
    failure_rate: float,
    num_turns: int,
    failure_threshold: int = 3,
    reset_timeout: float = 0.05,  # very short for demo
) -> dict:
    """Drive a FlakyProvider through a ProviderFallbackChain and collect stats."""
    primary = FlakyProvider(
        name="primary",
        model="claude-haiku-4-5",
        circuit_breaker=CircuitBreaker("primary", failure_threshold, reset_timeout),
        failure_rate=failure_rate,
    )
    fallback = FlakyProvider(
        name="fallback",
        model="gpt-4o-mini",
        circuit_breaker=CircuitBreaker("fallback", failure_threshold * 2, reset_timeout),
        failure_rate=0.0,  # fallback always works
    )
    chain = ProviderFallbackChain([primary, fallback])

    stats = {"success": 0, "primary_used": 0, "fallback_used": 0, "exhausted": 0}
    messages = [{"role": "user", "content": "test"}]

    for _ in range(num_turns):
        try:
            _, provider_used = chain.complete(messages)
            stats["success"] += 1
            if provider_used == "primary":
                stats["primary_used"] += 1
            else:
                stats["fallback_used"] += 1
            time.sleep(0.01)  # simulate inter-turn gap (allows circuit probing)
        except RuntimeError:
            stats["exhausted"] += 1

    return stats


print("=== Chaos test: 0% failure rate ===")
stats = run_chaos_test(failure_rate=0.0, num_turns=10)
print(f"  {stats}")
assert stats["exhausted"] == 0
assert stats["primary_used"] == 10

print("\n=== Chaos test: 80% failure rate ===")
stats = run_chaos_test(failure_rate=0.8, num_turns=20, failure_threshold=3)
print(f"  {stats}")
# Fallback should carry most turns once circuit trips
assert stats["exhausted"] == 0, f"No turns should be exhausted (fallback is reliable): {stats}"
print(f"  Fallback absorbed {stats['fallback_used']}/{stats['success']} turns — circuit breaker working ✓")

print("\n=== Chaos test: 100% failure (both providers down) ===")
primary = FlakyProvider(
    name="primary", model="m1",
    circuit_breaker=CircuitBreaker("p", failure_threshold=2, reset_timeout=60),
    failure_rate=1.0,
)
secondary = FlakyProvider(
    name="secondary", model="m2",
    circuit_breaker=CircuitBreaker("s", failure_threshold=2, reset_timeout=60),
    failure_rate=1.0,
)
bad_chain = ProviderFallbackChain([primary, secondary])
exhausted_count = 0
for _ in range(10):
    try:
        bad_chain.complete([{"role": "user", "content": "test"}])
    except RuntimeError:
        exhausted_count += 1
print(f"  Exhausted turns: {exhausted_count}/10 (expected once both circuits trip)")
```

## Trace: resilient provider call path

```
AIAgent.run_conversation(text)
    │
    ▼
ProviderFallbackChain.complete(messages)
    ├─ Entry 1: anthropic
    │     CircuitBreaker.call(entry.complete, messages)
    │       ├─ state == OPEN? → CircuitOpenError → skip to Entry 2
    │       ├─ state == CLOSED/HALF_OPEN → call provider
    │       │     ├─ success → _on_success() → state=CLOSED → return
    │       │     └─ ProviderError → _on_failure() → maybe OPEN → raise
    │       └─ ProviderError caught → skip to Entry 2
    │
    └─ Entry 2: openai
          CircuitBreaker.call(entry.complete, messages)
              └─ success → return (response, "openai")

Caller sees successful response — never aware of the primary failure.
```

## Hands-on exercises

1. **Wire CircuitBreaker into ProviderRegistry**: Wrap `ProviderRegistry.resolve()` so each registered provider has an associated `CircuitBreaker`. Verify the tutorial's `AIAgent` automatically skips tripped providers.

2. **Add `retry_with_backoff` to the agent loop**: Apply the `@retry_with_backoff(retryable=(ConnectionError,), max_retries=3)` decorator to `provider.complete_with_tools()` in `agent.py`. Run a chat session and observe the retry behaviour.

3. **DLQ replay integration**: Modify `CronScheduler.tick()` to push failed jobs to `DeadLetterQueue`. Add a CLI command `harness-agent cron replay <job_id>` that reads from the DLQ and re-dispatches.

4. **HALF_OPEN probe timing**: Modify the `CircuitBreaker` demo to use `reset_timeout=5` (5 seconds). Trip the circuit, wait 5 seconds, then send a succeeding call. Verify the state transitions: `OPEN → HALF_OPEN → CLOSED`.

5. **Chaos test parametrization**: Run `run_chaos_test` with failure rates of 0%, 25%, 50%, 75%, 100%. Plot `primary_used` vs `fallback_used` as a function of failure rate (use `matplotlib` or just print a table).

6. **Idempotency test**: Push the same `job_id` to the DLQ 10 times with different errors. Verify there is only one row in the DLQ but `attempts=10`.

## Common pitfalls

| Pitfall | Symptom | Diagnosis |
|---------|---------|----------|
| Retrying non-transient errors | `ValueError: invalid API key` retried 3 times | Pass `retryable=(ConnectionError, TimeoutError)` — never retry auth or validation errors |
| No jitter on retries | All clients retry at the same time — amplifies load | Add `random.uniform(0, delay * 0.1)` jitter to every retry delay |
| Circuit never resets | `reset_timeout` too high — system stays degraded | Set `reset_timeout` based on your provider's typical recovery time (30–60s) |
| `HALF_OPEN` probe with failing call | Circuit stays OPEN forever | Probe with a lightweight call (e.g. count tokens) not a full generation |
| DLQ without job_id | Same job pushed twice — double execution on replay | Always include a deterministic `job_id` in the job payload |
| Fallback on all exceptions | Fallback triggered by malformed input | Only fall back on `ProviderError` / `CircuitOpenError` — not `ValueError` from bad input |
| CircuitBreaker shared across threads without lock | Race condition on `_failure_count` | Use a `threading.Lock` inside `call()` — already in the implementation above |

## Checkpoint questions

1. Draw the circuit breaker state machine. What triggers each transition?
2. Why does exponential backoff without jitter cause a **thundering herd**? How does jitter fix it?
3. In `ProviderFallbackChain`, which exception causes a skip to the next provider? Which exceptions propagate to the caller?
4. What makes the `DeadLetterQueue.push()` idempotent? What SQL construct enables this?
5. A `HALF_OPEN` probe fails. What happens to the circuit state?
6. Why should `retryable` never include `ValueError` or `AuthenticationError`?
7. Your primary provider has `failure_threshold=3` and `reset_timeout=30s`. It fails 4 times in 10 seconds. Describe the exact sequence of states and what the caller experiences.

## Summary

| Concept | Key detail |
|---------|----------|
| `CircuitBreaker` | State machine: CLOSED → OPEN (on threshold failures) → HALF_OPEN (after reset_timeout) → CLOSED (on probe success) |
| `CircuitOpenError` | Raised immediately when circuit is OPEN — no network call |
| `retry_with_backoff` | Exponential backoff + jitter; `retryable` tuple controls which errors are retried |
| Jitter | Random noise on delay — prevents thundering herd when many clients retry simultaneously |
| `ProviderFallbackChain` | Tries providers in order; skips on `CircuitOpenError`; falls back on `ProviderError` |
| `DeadLetterQueue` | SQLite-backed; idempotent on `job_id`; supports `push` / `list_failed` / `replay` |
| `FlakyProvider` | Chaos injection — deterministic failure rate for resilience testing |
| Chaos harness | Drive the system with controlled failure rates; verify zero exhausted turns when fallback is available |

---

**Congratulations** — you have completed the production track of the Harness Agent tutorial.

| Chapter | Theme |
|---------|------|
| ch23 | Observability — structured logs, traces, metrics, health probes |
| ch24 | Scalability — multi-tenant isolation, rate limiting, shared session routing |
| ch25 | Reliability — circuit breakers, retry, fallback chains, dead-letter queue |

Together, these three pillars (**observability**, **scalability**, **reliability**) are the foundation for running an agent harness in production at any scale.
