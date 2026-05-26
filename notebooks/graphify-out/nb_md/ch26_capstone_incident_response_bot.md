# ch26_capstone_incident_response_bot

# Capstone A — Production Incident Response Bot

Harness Agent tutorial — `ch26_capstone_incident_response_bot.ipynb`

## Chapter objectives

- Wire **all 25 prior chapters** into a single, coherent SRE application.
- Build a `LogGenerator` that produces realistic service logs with configurable error injection.
- Implement an `IncidentPoller` that detects anomalies and fires investigations using a `CronScheduler`-like pattern.
- Orchestrate a **multi-agent investigation** (researcher → coder → planner) using `MultiAgentOrchestrator`.
- Instrument every subsystem with the full **observability triad**: structured logs, traces, and metrics from ch23.
- Layer in **reliability** from ch25: `ProviderFallbackChain`, `CircuitBreaker`, and `DeadLetterQueue`.
- Apply **multi-tenant isolation** from ch24: per-team `TenantContext`, `RateLimiter`, and `SessionRouter`.
- Close the loop with the **learning loop** (runbook authoring) and `GatewayRunner` (manual escalation).

## Prerequisites

ch00–ch25 completed or package installed. All code cells run in **simulation mode** — no real API key required.
Set `ANTHROPIC_API_KEY` (or any supported provider key) to enable live agent calls in the integration demo.

## Concept: architecture — all subsystems converge

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Incident Response Bot (ch26)                       │
│                                                                     │
│  ┌──────────────┐   anomaly    ┌───────────────────────────────┐   │
│  │ LogGenerator │ ──────────► │       IncidentPoller          │   │
│  │  (ch14 cron) │             │  (threshold → incident_id)    │   │
│  └──────────────┘             └───────────┬───────────────────┘   │
│                                           │ fires                  │
│                               ┌───────────▼───────────────────┐   │
│  ┌─────────────────────┐      │  InvestigationOrchestrator    │   │
│  │  TenantContext      │◄────►│  (MultiAgentOrchestrator)     │   │
│  │  RateLimiter        │      │  researcher → coder → planner │   │
│  │  SessionRouter      │      └───────────┬───────────────────┘   │
│  └─────────────────────┘                  │ wraps                  │
│       ch24 multi-tenant                   │                        │
│                               ┌───────────▼───────────────────┐   │
│  ┌─────────────────────┐      │  ProviderFallbackChain        │   │
│  │  StructuredLogger   │◄────►│  CircuitBreaker               │   │
│  │  MetricsCollector   │      │  DeadLetterQueue              │   │
│  │  instrument_turn    │      └───────────┬───────────────────┘   │
│  └─────────────────────┘                  │ on success             │
│       ch23 observability                  │                        │
│                               ┌───────────▼───────────────────┐   │
│  ┌─────────────────────┐      │  LearningLoop → SkillCatalog  │   │
│  │  GatewayRunner      │◄────►│  runbook authored to          │   │
│  │  (P0 escalation)    │      │  labs/skills/<incident>/      │   │
│  └─────────────────────┘      └───────────┬───────────────────┘   │
│       ch15 gateway                        │ archives               │
│                               ┌───────────▼───────────────────┐   │
│                               │  SessionStore + Trajectories  │   │
│                               │  (incident history + export)  │   │
│                               └───────────────────────────────┘   │
│                                    ch06 + ch19                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Concept: subsystem role table

| Subsystem used | Key class / call | Taught in |
|---|---|---|
| Agent loop | `AIAgent.run_conversation()` | ch03 |
| Tool registry | `get_registry()` | ch04 |
| Session store | `SessionStore.search()` | ch06 |
| Prompt assembly | `PromptBuilder` | ch07 |
| Skills | `SkillCatalog.discover()` | ch08 |
| Learning loop | `LearningLoop` | ch10 |
| Subagents | `AIAgent(isolated=True)` | ch12 |
| Cron scheduler | `CronScheduler.tick()` | ch14 |
| Gateway | `GatewayRunner.handle_message()` | ch15 |
| Multi-agent | `MultiAgentOrchestrator` | ch22 |
| Observability | `StructuredLogger`, `instrument_turn`, `MetricsCollector` | ch23 |
| Multi-tenancy | `TenantContext`, `RateLimiter`, `SessionRouter` | ch24 |
| Reliability | `CircuitBreaker`, `ProviderFallbackChain`, `DeadLetterQueue` | ch25 |
| Trajectory export | `export_trajectories()` | ch19 |

## Part 1: Simulated service log generator

```python
import os, time, uuid, random
from pathlib import Path
from dataclasses import dataclass, field
os.environ.setdefault('HARNESS_AGENT_HOME', 'labs')

LOG_TEMPLATES = [
    "INFO  app.server   Request processed in {ms}ms  path=/api/v1/users",
    "INFO  app.cache    Cache hit ratio {ratio:.2f}  keys=12843",
    "DEBUG app.auth     JWT validated  user_id=usr_{uid}",
    "INFO  app.db       Query OK  rows={rows}  latency={ms}ms",
    "WARN  app.server   Slow response {ms}ms > 2000ms threshold",
    "ERROR app.db       Connection pool exhausted  pool_size=20  waiting={wait}",
    "ERROR app.server   Unhandled exception: {exc}  trace_id={tid}",
    "ERROR app.cache    Redis timeout after 5000ms  host=cache-01",
    "ERROR app.disk     Disk usage {pct}% on /var/data — write failed",
]

ERROR_LINES = [t for t in LOG_TEMPLATES if t.startswith("ERROR")]
INFO_LINES  = [t for t in LOG_TEMPLATES if not t.startswith("ERROR")]

EXCEPTIONS = ["NullPointerException", "OutOfMemoryError", "SocketTimeoutException"]


@dataclass
class LogGenerator:
    log_path: Path
    error_rate: float = 0.15   # fraction of lines that are ERRORs

    def __post_init__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _render(self, template: str) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = template.format(
            ms=random.randint(12, 8000),
            ratio=random.uniform(0.6, 0.99),
            uid=random.randint(1000, 9999),
            rows=random.randint(1, 500),
            wait=random.randint(1, 50),
            exc=random.choice(EXCEPTIONS),
            tid=uuid.uuid4().hex[:8],
            pct=random.randint(85, 99),
        )
        return f"{ts} {line}"

    def emit_batch(self, n: int = 20) -> int:
        """Write n log lines; return count of ERROR lines injected."""
        error_count = 0
        with self.log_path.open("a") as fh:
            for _ in range(n):
                if random.random() < self.error_rate:
                    line = self._render(random.choice(ERROR_LINES))
                    error_count += 1
                else:
                    line = self._render(random.choice(INFO_LINES))
                fh.write(line + "\n")
        return error_count

    def tail(self, n: int = 10) -> list[str]:
        """Return last n lines from the log file."""
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text().splitlines()
        return lines[-n:]


# Demo: generate a batch and show the tail
log_path = Path("labs/services/app.log")
gen = LogGenerator(log_path, error_rate=0.20)
errors = gen.emit_batch(n=30)
print(f"Emitted 30 lines, {errors} ERROR(s) injected.")
print(f"Log file: {log_path}")
print("\n--- Last 8 lines ---")
for line in gen.tail(8):
    print(" ", line)
```

## Part 2: Incident detection — IncidentPoller

```python
import re, json, sqlite3
from typing import Optional

# ── Re-define DeadLetterQueue inline (origin: ch25) ─────────────────────────
@dataclass
class FailedJob:
    job_id: str
    payload: dict
    error: str
    failed_at: float
    attempts: int


class DeadLetterQueue:
    def __init__(self, db_path: Optional[Path] = None):
        home = Path(os.environ.get("HARNESS_AGENT_HOME", "labs"))
        self.db_path = db_path or (home / "cron" / "dlq.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dlq (
                    job_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    error TEXT NOT NULL, failed_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 1
                )
            """)

    def push(self, job_id: str, payload: dict, error: str) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO dlq (job_id, payload, error, failed_at, attempts) VALUES (?,?,?,?,1)
                ON CONFLICT(job_id) DO UPDATE SET
                    error=excluded.error, failed_at=excluded.failed_at, attempts=dlq.attempts+1
            """, (job_id, json.dumps(payload), error, time.time()))

    def list_failed(self) -> list:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM dlq ORDER BY failed_at DESC").fetchall()
        return [FailedJob(r["job_id"], json.loads(r["payload"]), r["error"], r["failed_at"], r["attempts"]) for r in rows]

    def clear(self) -> int:
        with self._connect() as conn:
            return conn.execute("DELETE FROM dlq").rowcount


# ── IncidentPoller ───────────────────────────────────────────────────────────
@dataclass
class IncidentPoller:
    """
    CronScheduler-like class that monitors a log file for ERROR bursts.
    When the error count in the last `window` lines exceeds `threshold`,
    it returns an incident_id and the offending lines.
    """
    log_path: Path
    threshold: int = 3
    window: int = 30              # lines to scan per poll
    dlq: DeadLetterQueue = field(default_factory=DeadLetterQueue)
    _last_line: int = field(default=0, init=False)

    def poll(self) -> tuple[Optional[str], list[str]]:
        """Returns (incident_id, error_lines) if threshold exceeded, else (None, [])."""
        if not self.log_path.exists():
            return None, []
        all_lines = self.log_path.read_text().splitlines()
        new_lines = all_lines[self._last_line:]
        self._last_line = len(all_lines)
        scan = new_lines[-self.window:] if len(new_lines) > self.window else new_lines
        error_lines = [l for l in scan if "ERROR" in l]
        if len(error_lines) >= self.threshold:
            incident_id = f"INC-{uuid.uuid4().hex[:6].upper()}"
            return incident_id, error_lines
        return None, []


# Demo
dlq = DeadLetterQueue()
poller = IncidentPoller(log_path, threshold=3, dlq=dlq)

# Generate a high-error batch to trigger detection
gen2 = LogGenerator(log_path, error_rate=0.50)
errors = gen2.emit_batch(n=20)
print(f"Injected {errors} additional ERRORs into log.")

incident_id, error_lines = poller.poll()
if incident_id:
    print(f"\nIncident detected: {incident_id}")
    print(f"Error lines ({len(error_lines)}):")
    for line in error_lines[:5]:
        print(" ", line)
else:
    print("No incident threshold exceeded on this poll.")
```

## Part 3: Multi-agent investigation — InvestigationOrchestrator

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class IncidentReport:
    incident_id: str
    root_cause: str
    steps_taken: list[str]
    runbook: str
    session_id: str


class InvestigationOrchestrator:
    """
    Wraps MultiAgentOrchestrator (ch22) for incident investigation.
    Three roles:
      researcher — reads and classifies error lines
      coder      — runs diagnostic analysis
      planner    — drafts remediation steps and runbook

    Simulation mode: returns deterministic results without a real API key.
    Live mode: set ANTHROPIC_API_KEY and uncomment the MultiAgentOrchestrator block.
    """

    def __init__(self, simulation: bool = True):
        self.simulation = simulation

    def _classify_errors(self, error_lines: list[str]) -> str:
        """Simple heuristic classification for simulation mode."""
        text = " ".join(error_lines).lower()
        if "pool exhausted" in text or "connection" in text:
            return "database_connection_saturation"
        if "disk" in text or "write failed" in text:
            return "disk_full"
        if "redis" in text or "timeout" in text:
            return "cache_layer_timeout"
        return "unknown_application_error"

    def investigate(self, incident_id: str, error_lines: list[str]) -> IncidentReport:
        """Returns a structured IncidentReport."""
        session_id = str(uuid.uuid4())

        if self.simulation:
            root_cause = self._classify_errors(error_lines)
            steps = [
                f"researcher: classified {len(error_lines)} error lines as '{root_cause}'",
                f"coder: ran diagnostic — top error frequency analysis complete",
                f"planner: drafted remediation for '{root_cause}'",
            ]
            runbook = (
                f"# Runbook: {root_cause.replace('_', ' ').title()}\n\n"
                f"**Incident**: {incident_id}\n\n"
                f"## Symptoms\n- {len(error_lines)} ERROR lines detected in window\n\n"
                f"## Root Cause\n{root_cause}\n\n"
                f"## Remediation Steps\n"
                f"1. Check application dashboard\n"
                f"2. Scale affected service or clear resource\n"
                f"3. Verify metrics return to baseline\n"
                f"4. Document in post-mortem\n"
            )
        else:
            # Live mode: uncomment to use MultiAgentOrchestrator
            # from harness_agent.graph.orchestrator import MultiAgentOrchestrator
            # orc = MultiAgentOrchestrator(roles=["researcher", "coder", "planner"])
            # result = orc.run(f"Incident {incident_id}: analyse errors: {error_lines}")
            # root_cause = result.root_cause
            # steps = result.steps
            # runbook = result.runbook
            raise RuntimeError("Set simulation=True or provide API key for live mode.")

        return IncidentReport(
            incident_id=incident_id,
            root_cause=root_cause,
            steps_taken=steps,
            runbook=runbook,
            session_id=session_id,
        )


# Demo: investigate the incident from Part 2
if incident_id:  # from previous cell
    orc = InvestigationOrchestrator(simulation=True)
    report = orc.investigate(incident_id, error_lines)
    print(f"Incident:   {report.incident_id}")
    print(f"Root cause: {report.root_cause}")
    print(f"Session ID: {report.session_id}")
    print("\nSteps taken:")
    for step in report.steps_taken:
        print(f"  - {step}")
    print("\nRunbook preview (first 5 lines):")
    for line in report.runbook.splitlines()[:5]:
        print(" ", line)
else:
    # Create a synthetic incident for demo
    incident_id = "INC-DEMO01"
    error_lines = ["ERROR app.db Connection pool exhausted"] * 5
    orc = InvestigationOrchestrator(simulation=True)
    report = orc.investigate(incident_id, error_lines)
    print(f"Demo incident: {report.incident_id}  root_cause={report.root_cause}")
```

## Part 4: Observability layer — instrument every investigation turn

```python
import contextlib
from typing import Any

# ── Re-define observability primitives inline (origin: ch23) ─────────────────
@dataclass
class StructuredLogger:
    name: str
    _context: dict = field(default_factory=dict)

    def bind(self, **kwargs) -> "StructuredLogger":
        child = StructuredLogger(self.name)
        child._context = {**self._context, **kwargs}
        return child

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        record = {"ts": round(time.time(), 3), "level": level, "logger": self.name,
                  "event": event, **self._context, **fields}
        print(json.dumps(record))

    def info(self, event: str, **fields):    self._emit("INFO",  event, **fields)
    def warning(self, event: str, **fields): self._emit("WARN",  event, **fields)
    def error(self, event: str, **fields):   self._emit("ERROR", event, **fields)


class _Noop:
    def labels(self, **_): return self
    def observe(self, _):  pass
    def inc(self, _=1):    pass


class MetricsCollector:
    def __init__(self):
        try:
            from prometheus_client import Counter, Histogram, CollectorRegistry
            self._reg = CollectorRegistry()
            self.turn_latency = Histogram("harness_turn_latency_seconds", "Turn latency",
                                          ["provider", "model"], registry=self._reg)
            self.tokens  = Counter("harness_token_count_total", "Tokens",
                                   ["provider", "direction"], registry=self._reg)
            self.errors  = Counter("harness_errors_total", "Errors",
                                   ["error_type"], registry=self._reg)
            self.incidents = Counter("harness_incidents_total", "Incidents detected",
                                     ["root_cause"], registry=self._reg)
        except ImportError:
            self.turn_latency = self.tokens = self.errors = self.incidents = _Noop()


@contextlib.contextmanager
def instrument_turn(logger: StructuredLogger, metrics: MetricsCollector,
                    session_id: str, provider: str = "simulation", model: str = "n/a"):
    """Context manager: logs + metrics around a single agent/investigation turn."""
    turn_log = logger.bind(session_id=session_id, provider=provider, model=model)
    t0 = time.perf_counter()
    turn_log.info("turn_start")
    try:
        yield turn_log
        latency = time.perf_counter() - t0
        metrics.turn_latency.labels(provider=provider, model=model).observe(latency)
        turn_log.info("turn_complete", latency_ms=round(latency * 1000, 1))
    except Exception as exc:
        latency = time.perf_counter() - t0
        metrics.errors.labels(error_type=type(exc).__name__).inc()
        turn_log.error("turn_failed", latency_ms=round(latency * 1000, 1), error=str(exc))
        raise


# Demo: wrap an investigation with full instrumentation
logger = StructuredLogger("incident_bot")
metrics = MetricsCollector()

test_incident_id = "INC-OBS01"
test_error_lines = ["ERROR app.db Connection pool exhausted"] * 4

print("=== Instrumented investigation ===\n")
logger.bind(incident_id=test_incident_id).info("incident_detected",
    error_count=len(test_error_lines), threshold=poller.threshold)

with instrument_turn(logger, metrics, session_id=str(uuid.uuid4())[:8],
                     provider="simulation") as turn_log:
    turn_log.info("investigation_start", incident_id=test_incident_id)
    time.sleep(0.02)  # simulate investigation latency
    orc2 = InvestigationOrchestrator(simulation=True)
    rep2 = orc2.investigate(test_incident_id, test_error_lines)
    metrics.incidents.labels(root_cause=rep2.root_cause).inc()
    turn_log.info("investigation_complete",
                  root_cause=rep2.root_cause, steps=len(rep2.steps_taken))
```

## Part 5: Reliability layer — circuit breakers and dead-letter queue

```python
from enum import Enum, auto
from threading import Lock

# ── Re-define CircuitBreaker inline (origin: ch25) ───────────────────────────
class State(Enum):
    CLOSED = auto()
    OPEN   = auto()
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
    def state(self): return self._state

    def call(self, fn, *args, **kwargs):
        with self._lock:
            if self._state == State.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.reset_timeout:
                    self._state = State.HALF_OPEN
                else:
                    raise CircuitOpenError(f"{self.name} OPEN")
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


# ── ProviderFallbackChain inline (origin: ch25) ──────────────────────────────
@dataclass
class ProviderEntry:
    name: str
    circuit_breaker: CircuitBreaker
    failure_rate: float = 0.0
    _calls: int = field(default=0, init=False)

    def complete(self, prompt: str) -> str:
        self._calls += 1
        if random.random() < self.failure_rate:
            raise ConnectionError(f"{self.name} injected failure #{self._calls}")
        return f"[{self.name}] Investigation response #{self._calls}"


class ProviderFallbackChain:
    def __init__(self, providers: list[ProviderEntry]):
        self._providers = providers

    def complete(self, prompt: str) -> tuple[str, str]:
        for entry in self._providers:
            try:
                response = entry.circuit_breaker.call(entry.complete, prompt)
                return response, entry.name
            except CircuitOpenError:
                print(f"    [fallback] {entry.name} circuit OPEN — skip")
            except Exception as e:
                print(f"    [fallback] {entry.name} failed: {e}")
        raise RuntimeError("All providers exhausted")


# Demo: investigation wrapped in reliability layer
primary   = ProviderEntry("anthropic", CircuitBreaker("anthropic", failure_threshold=2), failure_rate=1.0)
secondary = ProviderEntry("openai",    CircuitBreaker("openai",    failure_threshold=3), failure_rate=0.0)
chain = ProviderFallbackChain([primary, secondary])

dlq2 = DeadLetterQueue()
dlq2.clear()

print("=== Reliability-wrapped investigation (3 incidents) ===\n")
for i in range(3):
    inc_id = f"INC-REL0{i+1}"
    try:
        response, provider_used = chain.complete(f"Investigate {inc_id}")
        print(f"  {inc_id}: OK via {provider_used}")
    except RuntimeError as exc:
        dlq2.push(inc_id, {"incident_id": inc_id}, str(exc))
        print(f"  {inc_id}: FAILED → pushed to DLQ")

print(f"\nDLQ entries: {len(dlq2.list_failed())}")
print(f"primary circuit state: {primary.circuit_breaker.state.name}")
```

## Part 6: Multi-tenant isolation — per-team incident handling

```python
# ── Re-define TenantContext, RateLimiter, SessionRouter inline (origin: ch24) ─
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

    def ensure_dirs(self) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TokenBucket:
    capacity: int
    refill_rate: float
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    def try_acquire(self, n: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(float(self.capacity),
                               self._tokens + (now - self._last_refill) * self.refill_rate)
            self._last_refill = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


class RateLimiter:
    def __init__(self, capacity: int = 5, refill_rate: float = 1.0):
        self._capacity = capacity
        self._rate = refill_rate
        self._buckets: dict[str, TokenBucket] = {}

    def try_acquire(self, tenant_id: str) -> bool:
        if tenant_id not in self._buckets:
            self._buckets[tenant_id] = TokenBucket(self._capacity, self._rate)
        return self._buckets[tenant_id].try_acquire()


class InMemorySessionStore:
    def __init__(self):
        self._store: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def set(self, key: str, session_id: str, ttl_seconds: int = 3600) -> None:
        self._store[key] = session_id

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


class SessionRouter:
    def __init__(self, store: InMemorySessionStore):
        self._store = store

    def get_or_create(self, routing_key: str) -> tuple[str, bool]:
        existing = self._store.get(routing_key)
        if existing:
            return existing, False
        session_id = str(uuid.uuid4())
        self._store.set(routing_key, session_id)
        return session_id, True


# Demo: three teams, each with isolated tenant context + rate limiting
base_home = Path(os.environ.get("HARNESS_AGENT_HOME", "labs"))
limiter = RateLimiter(capacity=3, refill_rate=1.0)
router  = SessionRouter(InMemorySessionStore())

TEAMS = ["platform-ops", "backend-team", "frontend-team"]
incidents_per_team = {
    "platform-ops":   ["INC-P01", "INC-P02", "INC-P03", "INC-P04"],  # exceeds limit
    "backend-team":   ["INC-B01", "INC-B02"],
    "frontend-team":  ["INC-F01"],
}

print("=== Multi-tenant incident routing ===\n")
for team in TEAMS:
    ctx = TenantContext(team, base_home)
    ctx.ensure_dirs()
    for inc_id in incidents_per_team[team]:
        if limiter.try_acquire(team):
            sid, is_new = router.get_or_create(f"{team}:{inc_id}")
            print(f"  [{team}] {inc_id}: ALLOWED  session={sid[:8]}... new={is_new}")
        else:
            print(f"  [{team}] {inc_id}: RATE LIMITED")
```

## Part 7: Learning loop — automated runbook authoring

```python
# ── Learning loop integration ────────────────────────────────────────────────
# In live mode this calls LearningLoop.maybe_write_skill() from ch10.
# In simulation mode we write the runbook to labs/skills/ directly.

def save_runbook_as_skill(report: IncidentReport, skills_dir: Path) -> Path:
    """
    Simulate LearningLoop.maybe_write_skill().
    Writes the investigation runbook to labs/skills/<root_cause>/SKILL.md.
    In live mode, LearningLoop reads the conversation messages and authors this
    automatically when tool_call_count >= threshold.
    """
    skill_dir = skills_dir / report.root_cause.replace(" ", "_")
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(report.runbook)
    return skill_path


# Live mode (uncomment to use the real LearningLoop):
# from harness_agent.learning.skill_writer import LearningLoop
# loop = LearningLoop()
# loop.maybe_write_skill(messages)  # messages = conversation from InvestigationOrchestrator

skills_dir = base_home / "skills"

# Use the report from Part 3 (or create a demo one)
demo_report = report if 'report' in dir() else orc.investigate("INC-LRN01",
    ["ERROR app.db Connection pool exhausted"] * 5)

skill_path = save_runbook_as_skill(demo_report, skills_dir)
print(f"Runbook saved: {skill_path}")

# Show catalog discovery
try:
    from harness_agent.skills.loader import SkillCatalog
    catalog = SkillCatalog(skills_dir=skills_dir)
    skills = catalog.discover()
    print(f"\nSkillCatalog found {len(skills)} skill(s):")
    for s in skills[:5]:
        print(f"  - {s.name}: {getattr(s, 'description', '(no description)')[:60]}")
except ImportError:
    # Simulation: just list files
    skill_files = list(skills_dir.rglob("SKILL.md"))
    print(f"\nSkill files in {skills_dir}: {len(skill_files)}")
    for sf in skill_files[:5]:
        print(f"  - {sf}")

print("\nRunbook content:")
print(skill_path.read_text())
```

## Part 8: Manual escalation webhook — GatewayRunner

```python
# ── GatewayRunner escalation demo ────────────────────────────────────────────
# Simulates GatewayRunner.handle_message() receiving a human escalation
# for a P0 incident, continuing within the same incident session.

class EscalationGateway:
    """
    Simulates GatewayRunner (ch15) for manual P0 escalations.
    In live mode: replace _handle_simulated with the real AIAgent call.
    """

    def __init__(self, router: SessionRouter, logger: StructuredLogger):
        self._router = router
        self._log = logger

    def handle_message(self, user_id: str, text: str, incident_id: str = "") -> dict:
        """Process an escalation message; returns the bot response dict."""
        routing_key = f"escalation:{user_id}"
        session_id, is_new = self._router.get_or_create(routing_key)

        turn_log = self._log.bind(session_id=session_id, user_id=user_id,
                                   incident_id=incident_id)
        turn_log.info("escalation_received", text=text[:80], new_session=is_new)

        # Simulation response — replace with agent.run_conversation(text, session_id)
        response_text = (
            f"P0 escalation acknowledged for {incident_id or 'unspecified incident'}. "
            f"Incident session {session_id[:8]} {'started' if is_new else 'continued'}. "
            f"On-call engineer paged. Runbook retrieved from skills catalog."
        )
        turn_log.info("escalation_response", response=response_text[:80])
        return {"session_id": session_id, "response": response_text, "new_session": is_new}


gw_logger = StructuredLogger("incident_gateway")
gw_router = SessionRouter(InMemorySessionStore())
gateway   = EscalationGateway(gw_router, gw_logger)

print("=== Manual escalation via GatewayRunner ===\n")

# First escalation — starts a new session
r1 = gateway.handle_message("oncall-alice", "P0 database down, all writes failing",
                              incident_id="INC-P0001")
print(f"\nResponse: {r1['response']}")

print()

# Second message from same user — continues the session
r2 = gateway.handle_message("oncall-alice", "Fixed. Disk was full on db-primary. Cleared.",
                              incident_id="INC-P0001")
print(f"\nResponse: {r2['response']}")

assert r1["session_id"] == r2["session_id"], "Session continuity broken!"
print(f"\nSession continuity verified: {r1['session_id'][:8]}... (same for both turns)")
```

## Part 9: Incident history search — SessionStore

```python
# ── SessionStore search — incident history ────────────────────────────────────
# In live mode, past investigations are stored as sessions via AIAgent.
# SessionStore.search() uses SQLite FTS5 to find relevant past incidents.

try:
    from harness_agent.sessions.store import SessionStore
    store = SessionStore()

    # Seed some synthetic incident history
    seed_sessions = [
        ("INC-HIST01", "disk full on /var/data — cleared 40GB of old logs"),
        ("INC-HIST02", "database connection pool exhausted — scaled pool to 50"),
        ("INC-HIST03", "redis timeout — restarted cache-01, latency normalised"),
        ("INC-HIST04", "disk usage 97% on /tmp — tmpfs cleared by cron"),
    ]
    for sid, content in seed_sessions:
        store.append_turn(sid, role="assistant", content=content)

    print("=== Incident history search ===\n")
    for query in ["disk full", "connection pool", "redis"]:
        results = store.search(query, limit=3)
        print(f"Query '{query}': {len(results)} result(s)")
        for r in results[:2]:
            print(f"  session={r.get('session_id', '?')[:12]}  "
                  f"content={str(r.get('content', ''))[:60]}...")
        print()
except ImportError:
    # Simulation: in-memory incident log
    print("SessionStore not available — simulation mode.")
    incident_log = [
        {"session_id": "INC-HIST01", "content": "disk full on /var/data"},
        {"session_id": "INC-HIST02", "content": "database connection pool exhausted"},
        {"session_id": "INC-HIST03", "content": "redis timeout — restarted cache-01"},
    ]
    query = "disk full"
    results = [r for r in incident_log if query in r["content"]]
    print(f"Query '{query}': {len(results)} match(es)")
    for r in results:
        print(f"  {r['session_id']}: {r['content']}")
```

## Part 10: Trajectory export — every investigation becomes a training example

```python
# ── Trajectory export for incident post-mortems ──────────────────────────────
import json as _json

try:
    from harness_agent.trajectories.export import export_trajectories
    out_path = Path("labs/incidents.jsonl")
    count = export_trajectories(out_path)
    print(f"Exported {count} incident session(s) to {out_path}")
    if out_path.exists() and out_path.stat().st_size > 0:
        lines = out_path.read_text().splitlines()
        print(f"JSONL lines: {len(lines)}")
        first = _json.loads(lines[0]) if lines else {}
        print(f"First record keys: {list(first.keys())}")
except ImportError:
    # Simulation: write synthetic ShareGPT-format trajectories
    out_path = Path("labs/incidents.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic_incidents = [
        {
            "id": f"INC-SIM0{i+1}",
            "conversations": [
                {"from": "system", "value": "You are an SRE incident response agent."},
                {"from": "human", "value": f"Investigate incident INC-SIM0{i+1}: {desc}"},
                {"from": "gpt",   "value": f"Root cause identified: {rc}. Runbook authored."},
            ]
        }
        for i, (desc, rc) in enumerate([
            ("connection pool exhausted", "database_connection_saturation"),
            ("disk usage 97%", "disk_full"),
            ("redis timeout 5000ms", "cache_layer_timeout"),
        ])
    ]

    with out_path.open("w") as fh:
        for inc in synthetic_incidents:
            fh.write(_json.dumps(inc) + "\n")

    print(f"Simulation: wrote {len(synthetic_incidents)} incident trajectories to {out_path}")
    for inc in synthetic_incidents:
        turns = len(inc["conversations"])
        print(f"  {inc['id']}: {turns} turns")
```

## End-to-end integration demo — all subsystems firing together

```python
# ── Full pipeline: log injection → detection → investigation → observability
#                  → runbook authoring → DLQ check ──────────────────────────

print("=" * 60)
print("INCIDENT RESPONSE BOT — End-to-End Integration Demo")
print("=" * 60)

# 1. Fresh log file
e2e_log = Path("labs/services/e2e_demo.log")
e2e_log.unlink(missing_ok=True)
gen_e2e = LogGenerator(e2e_log, error_rate=0.40)
poller_e2e = IncidentPoller(e2e_log, threshold=3)
orc_e2e = InvestigationOrchestrator(simulation=True)
logger_e2e = StructuredLogger("incident_bot_e2e")
metrics_e2e = MetricsCollector()
dlq_e2e = DeadLetterQueue(Path("labs/cron/e2e_dlq.db"))
dlq_e2e.clear()
skills_dir_e2e = Path("labs/skills")

# 2. Simulate 3 polling rounds
for round_num in range(1, 4):
    print(f"\n--- Poll round {round_num} ---")
    injected = gen_e2e.emit_batch(n=15)
    print(f"  LogGenerator: {injected} ERRORs injected")

    inc_id, err_lines = poller_e2e.poll()
    if not inc_id:
        print(f"  IncidentPoller: no incident (errors below threshold)")
        continue

    print(f"  IncidentPoller: {inc_id} detected ({len(err_lines)} error lines)")

    # 3. Instrumented investigation
    with instrument_turn(logger_e2e, metrics_e2e, session_id=inc_id) as tlog:
        tlog.info("incident_detected", error_count=len(err_lines))

        try:
            rep_e2e = orc_e2e.investigate(inc_id, err_lines)
            tlog.info("investigation_complete",
                      root_cause=rep_e2e.root_cause,
                      steps=len(rep_e2e.steps_taken))
            metrics_e2e.incidents.labels(root_cause=rep_e2e.root_cause).inc()

            # 4. Learning loop — author runbook
            sk_path = save_runbook_as_skill(rep_e2e, skills_dir_e2e)
            tlog.info("runbook_authored", path=str(sk_path))
            print(f"  Investigation: root_cause={rep_e2e.root_cause}  runbook={sk_path.name}")

        except Exception as exc:
            dlq_e2e.push(inc_id, {"incident_id": inc_id}, str(exc))
            print(f"  Investigation FAILED → DLQ: {exc}")

# 5. DLQ summary
failed = dlq_e2e.list_failed()
print(f"\nDLQ entries: {len(failed)}")

# 6. Skill catalog summary
skill_files = list(skills_dir_e2e.rglob("SKILL.md"))
print(f"Skills authored: {len(skill_files)}")
for sf in skill_files[:3]:
    print(f"  - {sf.parent.name}")

print("\n=== End-to-end demo complete ===")
```

## Architecture retrospective — which chapter built which component

| Component in this bot | Key class / call | Chapter |
|---|---|---|
| `LogGenerator` | writes `labs/services/app.log` | ch14 (cron pattern) |
| `IncidentPoller` | threshold detection + `DeadLetterQueue` | ch14 + ch25 |
| `InvestigationOrchestrator` | `MultiAgentOrchestrator` (researcher, coder, planner) | ch22 |
| `StructuredLogger` + `instrument_turn` | JSON logs + OTel spans | ch23 |
| `MetricsCollector` | `harness_incidents_total` counter | ch23 |
| `HealthChecker` | pre-flight for bot startup | ch23 |
| `TenantContext` | per-team `labs/tenants/<id>/` isolation | ch24 |
| `RateLimiter` | token bucket per team | ch24 |
| `SessionRouter` | multi-turn continuity across stateless processes | ch24 |
| `CircuitBreaker` | trips at 3 provider failures | ch25 |
| `ProviderFallbackChain` | anthropic → openai auto-fallback | ch25 |
| `DeadLetterQueue` | failed investigation jobs | ch25 |
| `LearningLoop` / `save_runbook_as_skill` | runbook SKILL.md authoring | ch10 |
| `SkillCatalog.discover()` | lists authored runbooks | ch08 |
| `EscalationGateway` | `GatewayRunner.handle_message()` | ch15 |
| `SessionStore.search()` | incident history full-text search | ch06 |
| `export_trajectories()` | ShareGPT JSONL for fine-tuning | ch19 |
| `AIAgent.run_conversation()` | core agent loop (live mode) | ch03 |
| `PromptBuilder` | skill injection for next investigation | ch07 |

## Hands-on exercises

1. **Real API key integration**: Set `ANTHROPIC_API_KEY`, change `InvestigationOrchestrator(simulation=False)`, and wire in the real `MultiAgentOrchestrator` from ch22. Run a live investigation on the generated log file.

2. **Adaptive threshold**: Modify `IncidentPoller` to use a rolling average error rate instead of a fixed count. Trip the incident when the rate exceeds 2 standard deviations above the 5-minute baseline.

3. **Slack escalation adapter**: Replace `EscalationGateway` with a Slack webhook adapter (use the `requests` library). Send the investigation report to a `#incidents` channel when `root_cause` is `disk_full` or `database_connection_saturation`.

4. **Per-team runbook catalog**: Modify `save_runbook_as_skill` to write runbooks into `TenantContext.skills_dir` instead of the global `labs/skills/`. Verify that platform-ops and backend-team runbooks are isolated.

5. **DLQ replay pipeline**: Add a `replay_dlq()` function that dequeues all entries from `DeadLetterQueue` and re-runs `InvestigationOrchestrator.investigate()` on each. Wire it into a `CronScheduler` job that runs every 30 minutes.

6. **Grafana dashboard**: Wire `MetricsCollector` to a real Prometheus `start_http_server(9090)`. Create a dashboard panel for `harness_incidents_total` grouped by `root_cause`.

7. **Circuit breaker probe**: Trip the primary provider circuit (set `failure_rate=1.0`). After `reset_timeout` seconds, send a succeeding probe call. Verify the state transitions `OPEN → HALF_OPEN → CLOSED` in logs.

8. **Post-mortem export**: After running the end-to-end demo, call `export_trajectories()` and load the JSONL into a pandas DataFrame. Compute average `turns` per incident by `root_cause`.

## Common pitfalls

| Pitfall | Symptom | Diagnosis |
|---|---|---|
| `HARNESS_AGENT_HOME` not namespaced per tenant | All teams share `labs/skills/` — runbooks leak across teams | Pass `ctx.home` to `SkillCatalog` and `SessionStore`; never rely solely on env var for multi-tenant |
| `IncidentPoller` scanning full log each poll | Slow polls on large logs; duplicate incident detection | Track `_last_line` offset and only scan new lines from the last poll position |
| Circuit breaker shared between tenants | One team's provider failures trip the breaker for all teams | Create one `CircuitBreaker` per tenant per provider |
| DLQ `job_id` not deterministic | Same incident pushed twice on retry — both entries kept | Use `incident_id` as `job_id`; `ON CONFLICT DO UPDATE` ensures idempotency |
| Runbook never authored | `LearningLoop.should_author_skill()` threshold not reached | Benchmark tasks are sized for ≥5 tool calls; in simulation, call `save_runbook_as_skill()` directly |
| `EscalationGateway` loses session on restart | Every restart creates a new session for the same user | Swap `InMemorySessionStore` for `SqliteSessionStore` or `RedisSessionStore` (ch24) |
| Trajectory export produces 0 entries | `incidents.jsonl` is empty | Sessions need at least one user + one assistant turn; seed with `store.append_turn()` first |

## Checkpoint questions

1. Trace the full data flow from a log ERROR line to a SKILL.md being written. Name every class and method involved in order.
2. Why does `IncidentPoller` store `_last_line`? What bug occurs without it?
3. The `ProviderFallbackChain` catches `CircuitOpenError` and `ProviderError` but not `ValueError`. Why is this the correct boundary?
4. Explain how `TenantContext.skills_dir` prevents runbook leakage between the `platform-ops` and `backend-team` tenants.
5. A P0 escalation from the same user should always continue the same investigation session. Which class and method enforce this? What happens if the gateway process restarts?
6. After a successful investigation, what format does `export_trajectories()` produce, and how is it used downstream?
7. Your primary provider trips its circuit breaker. List the exact sequence of states and the time required before the next successful call goes through, given `failure_threshold=3` and `reset_timeout=30`.

## Summary

| Concept | Key detail |
|---|---|
| `LogGenerator` | Writes realistic service logs with configurable `error_rate`; returns count of ERRORs injected per batch |
| `IncidentPoller` | Scans new log lines since last poll; fires when error count ≥ `threshold`; returns `incident_id` |
| `InvestigationOrchestrator` | Wraps `MultiAgentOrchestrator` (researcher → coder → planner); returns `IncidentReport` |
| `instrument_turn` | Single context manager wiring ch23 logs + traces + metrics around any investigation turn |
| `ProviderFallbackChain` + `CircuitBreaker` | Auto-fallback when primary provider trips; DLQ captures exhausted jobs |
| `TenantContext` + `RateLimiter` | Per-team directory isolation + token-bucket quota enforcement |
| `SessionRouter` | Maps escalation routing key → session_id; preserves multi-turn continuity across restarts when backed by persistent store |
| `save_runbook_as_skill` | Simulates `LearningLoop.maybe_write_skill()`; every investigation authors a SKILL.md |
| `EscalationGateway` | Simulates `GatewayRunner.handle_message()`; manual P0 escalations continue the incident session |
| `export_trajectories` | Every investigation session becomes a ShareGPT-format training example |

---

### Capstone A certification checklist

- [ ] `LogGenerator` writes to `labs/services/app.log`
- [ ] `IncidentPoller.poll()` returns an `incident_id` when error threshold exceeded
- [ ] `InvestigationOrchestrator.investigate()` returns a complete `IncidentReport`
- [ ] `instrument_turn` emits structured JSON logs for every investigation
- [ ] `ProviderFallbackChain` falls through to secondary when primary circuit trips
- [ ] Failed investigations land in `DeadLetterQueue`
- [ ] Three teams have isolated tenant directories under `labs/tenants/`
- [ ] Runbook written to `labs/skills/<root_cause>/SKILL.md` after investigation
- [ ] `EscalationGateway` continues the same session for the same user
- [ ] `export_trajectories()` writes at least one incident to `labs/incidents.jsonl`

**Congratulations — all 25 chapters converge here.** Proceed to ch27 for the self-improvement flywheel.
