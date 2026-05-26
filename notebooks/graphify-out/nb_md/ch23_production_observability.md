# ch23_production_observability

# Production Observability

Harness Agent tutorial — `ch23_production_observability.ipynb`

## Chapter objectives

- Instrument `AIAgent.run_conversation()` with **structured JSON logs** carrying `session_id`, `turn_id`, and `provider` as correlation fields.
- Emit **OpenTelemetry spans** for each agent turn and tool call so traces can be visualised in Jaeger or any OTLP-compatible backend.
- Expose a **Prometheus `/metrics` endpoint** with four harness-specific gauges/histograms: turn latency, token counts, tool call counts, and error counts.
- Add a `/healthz` **liveness probe** that checks provider reachability and SQLite writability — required by Kubernetes and any load balancer.
- Understand the observability triad (logs, traces, metrics) and why each pillar catches different failure modes.

## Prerequisites

ch00–ch21 completed or package installed. Optional for live backends: `pip install opentelemetry-sdk prometheus_client`.

## Concept: The observability triad

Production systems fail in ways that are invisible to the agent itself. The **observability triad** gives operators three independent lenses:

| Pillar | What it captures | Typical question answered |
|--------|-----------------|---------------------------|
| **Logs** | Discrete events with context | "What happened in session `abc` at 14:32?" |
| **Traces** | Causal chains across calls | "Why did this turn take 8 seconds?" |
| **Metrics** | Aggregated numeric signals | "Is p99 latency trending up over the last hour?" |

### Why structured logs?

Plain text logs are hard to query at scale. Structured logs emit **JSON objects** — every field is queryable:

```json
{"level": "INFO", "event": "turn_complete", "session_id": "abc123",
 "turn_id": 3, "provider": "anthropic", "latency_ms": 1240,
 "input_tokens": 512, "output_tokens": 128, "tool_calls": 2}
```

A log aggregator (Loki, CloudWatch, Datadog) can filter `provider="anthropic" AND latency_ms > 5000` instantly.

### Why distributed traces?

A single agent turn may involve: prompt assembly → provider call → tool dispatch → second provider call. A trace captures the full **causal tree** with wall-clock timing per span:

```
run_conversation  [0ms ─────────────────────── 1240ms]
  build_prompt    [0ms ── 12ms]
  provider_call   [12ms ─────────── 820ms]
  dispatch_tool   [820ms ──── 980ms]
  provider_call   [980ms ─────── 1240ms]
```

### Why Prometheus metrics?

Logs tell you *what* happened. Metrics tell you *how often* and *how fast* at aggregate scale. Four essential harness metrics:

| Metric | Type | Labels | Alerts when |
|--------|------|--------|-------------|
| `harness_turn_latency_seconds` | Histogram | `provider`, `model` | p99 > 10s |
| `harness_token_count_total` | Counter | `provider`, `direction` | cost spike |
| `harness_tool_calls_total` | Counter | `tool_name`, `status` | error rate > 5% |
| `harness_errors_total` | Counter | `error_type` | any spike |

### Health probes

Kubernetes (and any load balancer) needs a `/healthz` endpoint that returns `200 OK` when the agent is ready to serve traffic. A health check should verify:
1. SQLite is writable (not just present)
2. At least one provider is configured (API key present)
3. `HARNESS_AGENT_HOME` is accessible

**Do not** call the LLM in a health check — it introduces latency and cost.

## How it works — annotated source

### Structured logger

```python
# observability/logger.py

import json, logging, time
from dataclasses import dataclass, field
from typing import Any

@dataclass
class StructuredLogger:
    name: str
    _context: dict = field(default_factory=dict)

    def bind(self, **kwargs) -> "StructuredLogger":
        """Return a child logger with extra context fields."""
        child = StructuredLogger(self.name)
        child._context = {**self._context, **kwargs}   # immutable: new dict
        return child

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        record = {
            "ts": time.time(),
            "level": level,
            "logger": self.name,
            "event": event,
            **self._context,          # bound context fields
            **fields,                 # per-call fields
        }
        print(json.dumps(record))     # stdout → log aggregator in production

    def info(self, event: str, **fields):  self._emit("INFO",  event, **fields)
    def warning(self, event: str, **fields): self._emit("WARN", event, **fields)
    def error(self, event: str, **fields): self._emit("ERROR", event, **fields)
```

### OpenTelemetry span decorator

```python
# observability/tracing.py

from functools import wraps
try:
    from opentelemetry import trace
    _tracer = trace.get_tracer("harness_agent")
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

def traced(span_name: str):
    """Decorator: wraps a method in an OTel span when otel is installed."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not _OTEL_AVAILABLE:
                return fn(*args, **kwargs)      # no-op when otel absent
            with _tracer.start_as_current_span(span_name) as span:
                try:
                    result = fn(*args, **kwargs)
                    span.set_status(trace.StatusCode.OK)
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(trace.StatusCode.ERROR)
                    raise
        return wrapper
    return decorator
```

### Metrics collector

```python
# observability/metrics.py

try:
    from prometheus_client import Counter, Histogram, start_http_server
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

class MetricsCollector:
    def __init__(self):
        if _PROM_AVAILABLE:
            self.turn_latency = Histogram(
                "harness_turn_latency_seconds",
                "Agent turn wall-clock latency",
                ["provider", "model"],
                buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
            )
            self.tokens = Counter(
                "harness_token_count_total",
                "LLM tokens consumed",
                ["provider", "direction"],   # direction: input / output
            )
            self.tool_calls = Counter(
                "harness_tool_calls_total",
                "Tool invocations",
                ["tool_name", "status"],     # status: ok / error
            )
            self.errors = Counter(
                "harness_errors_total",
                "Agent errors by type",
                ["error_type"],
            )
        else:
            # Stub — all calls become no-ops
            self.turn_latency = self.tokens = self.tool_calls = self.errors = _Noop()

class _Noop:
    def labels(self, **_): return self
    def observe(self, _): pass
    def inc(self, _=1): pass
```

## Reference implementation map

| Harness Agent | Industry pattern | Purpose |
|--------------|-----------------|--------|
| `StructuredLogger` | `structlog`, `python-json-logger` | JSON log emission with context binding |
| `@traced` decorator | OpenTelemetry Python SDK | Span creation and propagation |
| `MetricsCollector` | `prometheus_client` | Pull-based metrics endpoint |
| `HealthChecker` | Kubernetes liveness/readiness probes | Load balancer integration |

In production: replace `print(json.dumps(...))` with `structlog` configured to write to stdout; configure an OTLP exporter to ship spans to Jaeger or Tempo; run `start_http_server(8000)` before the agent starts.

## Design choices

| Choice | Rationale |
|--------|-----------|
| JSON logs to stdout | 12-factor app principle — let the platform collect logs; no file rotation complexity |
| `bind()` returns new logger | Immutable context — no accidental cross-request field bleed |
| OTel as optional import | Tutorial runs without `opentelemetry-sdk`; production enables it with one install |
| Prometheus pull model | Simpler than push (no StatsD server); scrape interval decoupled from agent logic |
| `/healthz` checks SQLite write, not read | A DB that can be read but not written is unhealthy for an agent |
| No LLM call in health check | Health checks run every 10s; a live LLM call would cost ~$0.01/min and add latency |
| `_Noop` stub for missing prometheus | Code paths identical whether prometheus is installed or not — no `if` branches in business logic |

## Implementation walkthrough

```python
import os, json, time, uuid
from dataclasses import dataclass, field
from typing import Any
os.environ.setdefault('HARNESS_AGENT_HOME', 'labs')

# ── StructuredLogger ────────────────────────────────────────────────────────
@dataclass
class StructuredLogger:
    name: str
    _context: dict = field(default_factory=dict)

    def bind(self, **kwargs) -> "StructuredLogger":
        child = StructuredLogger(self.name)
        child._context = {**self._context, **kwargs}
        return child

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        record = {
            "ts": round(time.time(), 3),
            "level": level,
            "logger": self.name,
            "event": event,
            **self._context,
            **fields,
        }
        print(json.dumps(record))

    def info(self, event: str, **fields):    self._emit("INFO",  event, **fields)
    def warning(self, event: str, **fields): self._emit("WARN",  event, **fields)
    def error(self, event: str, **fields):   self._emit("ERROR", event, **fields)


# ── Demo: bind context, then emit events ────────────────────────────────────
root_log = StructuredLogger("harness_agent")

session_id = str(uuid.uuid4())[:8]
turn_log = root_log.bind(session_id=session_id, provider="anthropic", model="claude-haiku-4-5")

turn_log.info("turn_start", turn_id=1, input_tokens=256)
turn_log.info("tool_call",  tool="read_file", path="logs/app.log")
turn_log.info("turn_complete", turn_id=1, latency_ms=843, output_tokens=128, tool_calls=1)

# Error example — same context, different event
turn_log.error("provider_error", error_type="RateLimitError", retry_in_s=30)
```

```python
import time
from functools import wraps

# ── Tracing: no-op stub when opentelemetry is absent ────────────────────────
try:
    from opentelemetry import trace as otel_trace
    _tracer = otel_trace.get_tracer("harness_agent")
    _OTEL_AVAILABLE = True
    print("OpenTelemetry SDK found — real spans will be emitted.")
except ImportError:
    _OTEL_AVAILABLE = False
    print("OpenTelemetry SDK not installed — using no-op tracer.")
    print("Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp")


class _NoopSpan:
    """Drop-in replacement when OTel is absent."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def set_attribute(self, *_): pass
    def record_exception(self, _): pass
    def set_status(self, *_): pass


def start_span(name: str, **attrs):
    """Context manager: real OTel span or no-op."""
    if _OTEL_AVAILABLE:
        span = _tracer.start_as_current_span(name)
        return span
    return _NoopSpan()


# ── Demo: instrument a fake agent turn ──────────────────────────────────────
log = StructuredLogger("harness_agent").bind(session_id="demo-01")

t0 = time.perf_counter()
with start_span("run_conversation") as turn_span:
    log.info("turn_start", turn_id=1)

    with start_span("build_prompt") as ps:
        time.sleep(0.005)  # simulate prompt assembly
        log.info("prompt_built", tokens=320)

    with start_span("provider_call") as pc:
        time.sleep(0.020)  # simulate LLM round-trip
        log.info("provider_response", input_tokens=320, output_tokens=85)

    with start_span("dispatch_tool") as ts:
        time.sleep(0.003)  # simulate tool execution
        log.info("tool_complete", tool="read_file", status="ok")

latency_ms = round((time.perf_counter() - t0) * 1000, 1)
log.info("turn_complete", latency_ms=latency_ms)
print(f"\nTotal simulated latency: {latency_ms} ms")
```

```python
# ── Metrics: Prometheus or no-op stub ───────────────────────────────────────
try:
    from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest
    _PROM_AVAILABLE = True
    _reg = CollectorRegistry()  # isolated registry so cells can re-run without name conflicts
    print("prometheus_client found — real metrics.")
except ImportError:
    _PROM_AVAILABLE = False
    print("prometheus_client not installed — using _Noop stub.")
    print("Install with: pip install prometheus_client")


class _Noop:
    def labels(self, **_): return self
    def observe(self, _): pass
    def inc(self, _=1): pass


class MetricsCollector:
    def __init__(self):
        if _PROM_AVAILABLE:
            self.turn_latency = Histogram(
                "harness_turn_latency_seconds",
                "Agent turn wall-clock latency",
                ["provider", "model"],
                buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
                registry=_reg,
            )
            self.tokens = Counter(
                "harness_token_count_total",
                "LLM tokens consumed",
                ["provider", "direction"],
                registry=_reg,
            )
            self.tool_calls = Counter(
                "harness_tool_calls_total",
                "Tool invocations",
                ["tool_name", "status"],
                registry=_reg,
            )
            self.errors = Counter(
                "harness_errors_total",
                "Agent errors by type",
                ["error_type"],
                registry=_reg,
            )
        else:
            self.turn_latency = self.tokens = self.tool_calls = self.errors = _Noop()


# ── Demo: record some fake observations ─────────────────────────────────────
metrics = MetricsCollector()

metrics.turn_latency.labels(provider="anthropic", model="claude-haiku-4-5").observe(0.843)
metrics.tokens.labels(provider="anthropic", direction="input").inc(512)
metrics.tokens.labels(provider="anthropic", direction="output").inc(128)
metrics.tool_calls.labels(tool_name="read_file", status="ok").inc()
metrics.tool_calls.labels(tool_name="run_bash", status="error").inc()
metrics.errors.labels(error_type="ToolError").inc()

print("Metrics recorded.")

if _PROM_AVAILABLE:
    output = generate_latest(_reg).decode()
    # Show a filtered excerpt — the turn_latency histogram
    for line in output.splitlines():
        if line and not line.startswith("#") and "harness_turn" in line:
            print(line)
else:
    print("(prometheus_client not installed — install to see real metric output)")
```

```python
import sqlite3, tempfile, os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
os.environ.setdefault('HARNESS_AGENT_HOME', 'labs')

# ── HealthChecker ────────────────────────────────────────────────────────────
@dataclass
class HealthStatus:
    name: str
    ok: bool
    detail: str = ""


class HealthChecker:
    def __init__(self, home: Optional[str] = None):
        self.home = Path(home or os.environ.get("HARNESS_AGENT_HOME", "labs"))

    def check_home(self) -> HealthStatus:
        ok = self.home.exists() and os.access(self.home, os.W_OK)
        return HealthStatus("home", ok, str(self.home) if ok else f"{self.home} not writable")

    def check_sqlite(self) -> HealthStatus:
        """Verify SQLite is writable by doing a real write."""
        db_path = self.home / "sessions" / "health_probe.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE IF NOT EXISTS probe (ts REAL)")
            conn.execute("INSERT INTO probe VALUES (?)", (time.time(),))
            conn.commit()
            conn.close()
            return HealthStatus("sqlite", True, str(db_path))
        except Exception as exc:
            return HealthStatus("sqlite", False, str(exc))

    def check_provider(self) -> HealthStatus:
        """Check that at least one API key is present — no live call."""
        keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
        found = [k for k in keys if os.environ.get(k)]
        ok = len(found) > 0
        return HealthStatus("provider", ok, f"found: {found}" if ok else "no API key set")

    def run(self) -> list[HealthStatus]:
        return [self.check_home(), self.check_sqlite(), self.check_provider()]

    def is_healthy(self) -> bool:
        return all(s.ok for s in self.run())


# ── Demo ─────────────────────────────────────────────────────────────────────
checker = HealthChecker()
statuses = checker.run()

print("=== /healthz ===")
for s in statuses:
    icon = "✓" if s.ok else "✗"
    print(f"  [{icon}] {s.name:<12} {s.detail}")

overall = "HEALTHY" if all(s.ok for s in statuses) else "UNHEALTHY"
print(f"\nStatus: {overall}")
```

```python
import time, contextlib

# ── End-to-end: instrument a mock agent turn with all three pillars ──────────
log = StructuredLogger("harness_agent")
metrics = MetricsCollector()


@contextlib.contextmanager
def instrument_turn(session_id: str, provider: str, model: str):
    """Context manager that logs + traces + records metrics for one agent turn."""
    turn_log = log.bind(session_id=session_id, provider=provider, model=model)
    t0 = time.perf_counter()
    turn_log.info("turn_start")
    try:
        with start_span("run_conversation"):
            yield turn_log
        latency = time.perf_counter() - t0
        metrics.turn_latency.labels(provider=provider, model=model).observe(latency)
        turn_log.info("turn_complete", latency_ms=round(latency * 1000, 1))
    except Exception as exc:
        latency = time.perf_counter() - t0
        error_type = type(exc).__name__
        metrics.errors.labels(error_type=error_type).inc()
        turn_log.error("turn_failed", latency_ms=round(latency * 1000, 1), error=str(exc))
        raise


# Successful turn
print("--- Successful turn ---")
with instrument_turn("sess-abc", "anthropic", "claude-haiku-4-5") as tlog:
    time.sleep(0.015)  # simulate work
    metrics.tokens.labels(provider="anthropic", direction="input").inc(400)
    metrics.tokens.labels(provider="anthropic", direction="output").inc(90)
    metrics.tool_calls.labels(tool_name="read_file", status="ok").inc()
    tlog.info("tool_call", tool="read_file", status="ok")

# Failed turn
print("\n--- Failed turn ---")
try:
    with instrument_turn("sess-xyz", "openai", "gpt-4o") as tlog:
        time.sleep(0.005)
        raise ConnectionError("Provider timeout after 5000ms")
except ConnectionError:
    pass  # error was already logged by instrument_turn
```

## Trace: request path with observability

```
User sends message
    │
    ▼
instrument_turn(session_id, provider, model)
    ├─ StructuredLogger.bind(session_id, provider, model)
    ├─ start_span("run_conversation")
    ├─ log.info("turn_start")
    │
    ├─ start_span("build_prompt") → log.info("prompt_built", tokens=N)
    ├─ start_span("provider_call") → log.info("provider_response", ...)
    ├─ start_span("dispatch_tool") → log.info("tool_complete", ...)
    │
    ├─ metrics.turn_latency.observe(latency)
    ├─ metrics.tokens.inc(input_tokens)
    └─ log.info("turn_complete", latency_ms=N)

All three pillars captured in one context manager — zero changes to AIAgent business logic.
```

## Hands-on exercises

1. **Add a `turn_id` counter**: Modify `StructuredLogger` so that each call to `info()` automatically increments a `turn_id` field stored in `_context`. Verify in the output.

2. **Wire up a real OTel exporter**: Install `opentelemetry-sdk opentelemetry-exporter-otlp` and add an OTLP exporter that ships spans to `http://localhost:4317`. Start a local Jaeger instance with Docker:
   ```bash
   docker run -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one
   ```
   Then run the `instrument_turn` cell and view the trace at `http://localhost:16686`.

3. **Expose Prometheus endpoint**: Add `from prometheus_client import start_http_server; start_http_server(8000)` before the metrics cell, then curl `http://localhost:8000/metrics` to see the raw exposition format.

4. **Failing health check**: Temporarily set `HARNESS_AGENT_HOME` to a path that doesn't exist and re-run `HealthChecker`. Verify the `home` check returns `ok=False`.

5. **Grafana dashboard**: Import this minimal dashboard JSON into a local Grafana instance (requires Prometheus running):
   ```json
   {"panels": [{"title": "Turn latency p99", "targets": [{"expr": "histogram_quantile(0.99, rate(harness_turn_latency_seconds_bucket[5m]))"}]}, {"title": "Token rate", "targets": [{"expr": "rate(harness_token_count_total[5m])"}]}]}
   ```

6. **Log correlation**: After running `instrument_turn`, grep the output for the `session_id` field. Confirm every log line for a given turn shares the same `session_id`.

## Common pitfalls

| Pitfall | Symptom | Diagnosis |
|---------|---------|----------|
| Mutating logger context | Fields bleed across sessions | Use `bind()` which creates a new dict — never mutate `_context` in place |
| Health check calls LLM | K8s probe times out | Move API key check to env var inspection only — no live calls |
| OTel exporter not configured | Spans silently dropped | Must call `TracerProvider` + exporter setup before `get_tracer()` |
| Metrics counter name collision | `ValueError` on cell re-run | Use isolated `CollectorRegistry()` per test; in production use a singleton |
| Histogram bucket mismatch | Observations all in last bucket | Set buckets to match expected latency range (0.1–30s for LLM calls) |
| Missing `session_id` in logs | Can't correlate logs to conversation | Bind `session_id` at the start of every turn, not just on errors |
| Log aggregator rejects non-JSON | Raw text lines ignored | Ensure every log emission goes through `json.dumps` — no `print()` for debug |

## Checkpoint questions

1. Name the three observability pillars. What failure mode does each catch that the others cannot?
2. Why does `StructuredLogger.bind()` return a new object instead of modifying `_context` in place?
3. What is the difference between a Prometheus Counter and a Histogram? Which would you use for turn latency?
4. Why should a `/healthz` health check never call the LLM provider?
5. Describe the `_Noop` pattern. Why is it preferable to `if prometheus_available:` branches throughout the code?
6. What does an OTel span capture that a log line does not?
7. A team reports that p99 latency spiked from 1s to 12s but error rates didn't change. Which observability pillar would help you diagnose this first?

## Summary

| Concept | Key detail |
|---------|----------|
| Structured logs | JSON objects with bound context (`session_id`, `provider`) — queryable at scale |
| `StructuredLogger.bind()` | Immutable context propagation — child loggers inherit and extend parent context |
| OTel spans | Causal chains with timing per span — instrument with `start_span()` context manager |
| `_Noop` stub | Business logic unchanged whether prometheus is installed or not |
| Core metrics | `turn_latency` (Histogram), `token_count` (Counter), `tool_calls` (Counter), `errors` (Counter) |
| `HealthChecker` | Checks home writable + SQLite writable + API key present — no live LLM calls |
| `instrument_turn` | One context manager wires all three pillars around a single agent turn |

**ch24** shows how to scale beyond a single SQLite file — multi-tenant namespacing, rate limiting, and a shared session store interface.
