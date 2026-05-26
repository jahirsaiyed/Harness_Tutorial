# ch27_capstone_self_improving_agent

# Capstone B — Self-Improving Agent Pipeline

Harness Agent tutorial — `ch27_capstone_self_improving_agent.ipynb`

## Chapter objectives

- Define a **10-task benchmark suite** that covers the full range of harness capabilities.
- Run a **baseline benchmark** and record pass rate, latency, and token counts per task.
- **Collect trajectories** from passing sessions and export them in ShareGPT format via `export_trajectories()`.
- **Extract reusable skills** from successful trajectories using `LearningLoop` and populate `SkillCatalog`.
- **Inject skills** into the next benchmark run via `PromptBuilder` and measure the improvement delta.
- Show the **fine-tuning pipeline setup** (LLaMA-Factory, Axolotl, OpenAI format) that would train a model on collected trajectories.
- Automate the full pipeline as a **nightly `CronScheduler` job** with `BenchmarkMetrics` trending over time.

## Prerequisites

ch00–ch26 completed or package installed. All benchmark runs use a **mock agent** — no real API key required.
Set `ANTHROPIC_API_KEY` to enable live agent runs in any cell marked `# LIVE MODE`.

## Concept: the self-improvement flywheel

```
                    ┌───────────────────────┐
                    │   1. Run Benchmark     │
                    │   (10 tasks, measure   │
                    │   pass rate / latency) │
                    └───────────┬───────────┘
                                │ passing sessions
                    ┌───────────▼───────────┐
                    │ 2. Collect Trajectories│
                    │  export_trajectories() │
                    │  → JSONL (ShareGPT)    │
                    └───────────┬───────────┘
                                │ high-quality examples
                    ┌───────────▼───────────┐
                    │ 3. Extract Skills      │
                    │  LearningLoop          │
                    │  → labs/skills/        │
                    └───────────┬───────────┘
                                │ discovered skills
                    ┌───────────▼───────────┐
                    │ 4. Inject & Re-run     │
                    │  PromptBuilder +       │
                    │  SkillCatalog          │
                    └───────────┬───────────┘
                                │ improved agent
                    ┌───────────▼───────────┐
                    │ 5. Measure & Repeat    │
                    │  BenchmarkMetrics      │
                    │  CronScheduler (nightly│
                    └───────────┴───────────┘
                                │
                          ◄─────┘  flywheel: each run
                                   makes the next cheaper
```

## Concept: what does "better" mean?

The flywheel produces measurable improvements across four dimensions:

| Metric | Definition | Improved by |
|---|---|---|
| **Pass rate** | fraction of tasks where `success_fn(response)` returns `True` | Skill injection gives the agent prior knowledge |
| **Latency** | wall-clock ms per task | Fewer retry turns needed — agent knows the pattern |
| **Token efficiency** | total `input + output` tokens per task | Shorter reasoning chains for familiar patterns |
| **Skill reuse rate** | fraction of tasks where an existing skill was matched | Grows as the skill catalog matures |

A good self-improvement loop shows: **pass_rate ↑, latency ↓, tokens ↓, skill_reuse ↑** across successive runs.

## Part 1: Benchmark definition

```python
import os, time, uuid, random, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
os.environ.setdefault('HARNESS_AGENT_HOME', 'labs')


@dataclass
class BenchmarkTask:
    task_id: str
    prompt: str
    success_fn: Callable[[str], bool]   # takes assistant response text, returns bool
    category: str = "general"


BENCHMARK_TASKS: list[BenchmarkTask] = [
    # Arithmetic
    BenchmarkTask(
        "calc-01", "What is 1234 * 5678? Reply with only the number.",
        lambda t: "7006652" in t.replace(",", "").replace(" ", ""),
        category="arithmetic",
    ),
    BenchmarkTask(
        "calc-02", "What is 17 ** 4? Reply with only the number.",
        lambda t: "83521" in t.replace(",", ""),
        category="arithmetic",
    ),
    # Code generation
    BenchmarkTask(
        "code-01", "Write a Python function `add(a, b)` that returns a + b. Include the def line.",
        lambda t: "def add" in t and "return" in t,
        category="code_generation",
    ),
    BenchmarkTask(
        "code-02", "Write a Python list comprehension that squares numbers 1–10.",
        lambda t: "[" in t and "**2" in t or "*x" in t.lower() or "square" in t.lower(),
        category="code_generation",
    ),
    # Multi-step reasoning
    BenchmarkTask(
        "reason-01",
        "If a train travels at 120 km/h for 2.5 hours, how many km does it cover? Show your reasoning.",
        lambda t: "300" in t,
        category="reasoning",
    ),
    BenchmarkTask(
        "reason-02",
        "List three root causes of database connection pool exhaustion. Number them 1, 2, 3.",
        lambda t: "1." in t and "2." in t and "3." in t,
        category="reasoning",
    ),
    # File/text manipulation
    BenchmarkTask(
        "text-01", "Reverse the string 'harness'. Reply with only the reversed string.",
        lambda t: "ssenwrah" in t.lower().replace(" ", ""),
        category="text_manipulation",
    ),
    BenchmarkTask(
        "text-02",
        "Convert 'hello_world_agent' from snake_case to CamelCase. Reply with only the result.",
        lambda t: "HelloWorldAgent" in t,
        category="text_manipulation",
    ),
    # Agent self-knowledge
    BenchmarkTask(
        "agent-01",
        "Name two subsystems of Harness Agent that are taught in chapters 14 and 15 respectively.",
        lambda t: ("cron" in t.lower() or "scheduler" in t.lower()) and
                  ("gateway" in t.lower() or "cli" in t.lower()),
        category="agent_knowledge",
    ),
    BenchmarkTask(
        "agent-02",
        "What SQLite extension does SessionStore use for full-text search? Name only the extension.",
        lambda t: "fts5" in t.lower() or "fts" in t.lower(),
        category="agent_knowledge",
    ),
]

assert len(BENCHMARK_TASKS) == 10, f"Expected 10 tasks, got {len(BENCHMARK_TASKS)}"

print(f"Benchmark suite: {len(BENCHMARK_TASKS)} tasks")
print()
categories = {}
for t in BENCHMARK_TASKS:
    categories.setdefault(t.category, []).append(t.task_id)
for cat, ids in categories.items():
    print(f"  {cat:<25} {ids}")
```

## Part 2: Baseline run — MockAgent

```python
# ── MockAgent: deterministic responses for simulation ────────────────────────
# Replace with AIAgent (ch03) for live runs.

MOCK_RESPONSES: dict[str, str] = {
    "calc-01":   "The answer is 7,006,652",
    "calc-02":   "83521",
    "code-01":   "def add(a, b):\n    return a + b",
    "code-02":   "squares = [x**2 for x in range(1, 11)]",
    "reason-01": "120 km/h × 2.5 h = 300 km",
    "reason-02": "1. Too many concurrent connections\n2. Connection leaks\n3. Slow queries holding connections",
    "text-01":   "ssenwrah",
    "text-02":   "The result is HelloWorldAgent",
    "agent-01":  "Chapter 14 covers the CronScheduler; chapter 15 covers the Gateway and CLI.",
    "agent-02":  "FTS5",
}

# Baseline pass rate is imperfect — agent-01 fails without skill injection
BASELINE_FAILURES = {"agent-01"}  # simulate knowledge gap closed by skill injection


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    session_id: str
    response: str = ""


class MockAgent:
    """Simulates AIAgent.run_conversation() without a real API call."""

    def __init__(self, skill_boost: set[str] = None):
        self._skill_boost = skill_boost or set()   # task_ids now passing due to skills

    def run_conversation(self, prompt: str, task_id: str = "") -> dict:
        time.sleep(random.uniform(0.01, 0.05))  # simulate latency
        failing = BASELINE_FAILURES - self._skill_boost
        response = "" if task_id in failing else MOCK_RESPONSES.get(task_id, "(no answer)")
        return {
            "assistant_text": response,
            "session_id": str(uuid.uuid4()),
            "input_tokens": len(prompt.split()) * 2,
            "output_tokens": len(response.split()) * 2,
        }


def run_benchmark(agent: MockAgent, tasks: list[BenchmarkTask]) -> list[TaskResult]:
    """Run all tasks; return one TaskResult per task."""
    results = []
    for task in tasks:
        t0 = time.perf_counter()
        out = agent.run_conversation(task.prompt, task_id=task.task_id)
        latency_ms = (time.perf_counter() - t0) * 1000
        passed = task.success_fn(out["assistant_text"])
        results.append(TaskResult(
            task_id=task.task_id,
            passed=passed,
            latency_ms=round(latency_ms, 1),
            input_tokens=out["input_tokens"],
            output_tokens=out["output_tokens"],
            session_id=out["session_id"],
            response=out["assistant_text"],
        ))
    return results


# LIVE MODE (uncomment to use real AIAgent):
# from harness_agent.agent import AIAgent
# live_agent = AIAgent()
# baseline_results = run_benchmark(live_agent, BENCHMARK_TASKS)

baseline_agent = MockAgent()
baseline_results = run_benchmark(baseline_agent, BENCHMARK_TASKS)

assert isinstance(baseline_results, list)
assert all(isinstance(r, TaskResult) for r in baseline_results)

passed  = sum(r.passed for r in baseline_results)
total   = len(baseline_results)
avg_lat = sum(r.latency_ms for r in baseline_results) / total
avg_tok = sum(r.input_tokens + r.output_tokens for r in baseline_results) / total

print(f"=== Baseline benchmark ({passed}/{total} passed, "
      f"{passed/total*100:.0f}% pass rate) ===")
print()
print(f"{'task_id':<15} {'passed':<8} {'latency_ms':>12} {'tokens':>8}")
print("-" * 48)
for r in baseline_results:
    tok = r.input_tokens + r.output_tokens
    mark = "✓" if r.passed else "✗"
    print(f"{r.task_id:<15} {mark:<8} {r.latency_ms:>12.1f} {tok:>8}")
print("-" * 48)
print(f"{'AVERAGE':<15} {'':8} {avg_lat:>12.1f} {avg_tok:>8.0f}")
```

## Part 3: Trajectory collection

```python
# ── Collect trajectories from passing sessions ───────────────────────────────

def collect_trajectories(
    results: list[TaskResult],
    tasks: list[BenchmarkTask],
    out_path: Path,
) -> Path:
    """
    Exports passing sessions to `out_path` in ShareGPT JSONL format.
    In live mode: call export_trajectories() on sessions stored by AIAgent.
    In simulation: construct the JSONL from TaskResult data.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    task_map = {t.task_id: t for t in tasks}
    passing = [r for r in results if r.passed]

    try:
        from harness_agent.trajectories.export import export_trajectories
        count = export_trajectories(out_path)
        print(f"export_trajectories(): {count} session(s) → {out_path}")
    except ImportError:
        # Simulation: write synthetic ShareGPT records
        with out_path.open("w") as fh:
            for r in passing:
                task = task_map[r.task_id]
                record = {
                    "id": r.session_id,
                    "task_id": r.task_id,
                    "category": task.category,
                    "conversations": [
                        {"from": "system",  "value": "You are Harness Agent."},
                        {"from": "human",   "value": task.prompt},
                        {"from": "gpt",     "value": r.response},
                    ],
                    "metadata": {
                        "latency_ms":    r.latency_ms,
                        "input_tokens":  r.input_tokens,
                        "output_tokens": r.output_tokens,
                    },
                }
                fh.write(json.dumps(record) + "\n")
        print(f"Simulation: wrote {len(passing)} passing trajectories → {out_path}")

    return out_path


traj_path = Path("labs/benchmark_trajectories.jsonl")
collect_trajectories(baseline_results, BENCHMARK_TASKS, traj_path)

# Inspect the JSONL
lines = traj_path.read_text().splitlines()
print(f"\nJSONL file: {traj_path}")
print(f"Records:    {len(lines)}")
if lines:
    sample = json.loads(lines[0])
    print(f"First record — task_id={sample['task_id']}  "
          f"turns={len(sample['conversations'])}  "
          f"category={sample['category']}")
```

```python
# ── Quality filter: keep only trajectories with ≥2 conversation turns ────────

all_records = [json.loads(l) for l in traj_path.read_text().splitlines() if l.strip()]
quality = [r for r in all_records if len(r["conversations"]) >= 2]

print(f"Total trajectories:   {len(all_records)}")
print(f"Quality examples:     {len(quality)}  (≥2 turns)")

# Category distribution
cat_counts: dict[str, int] = {}
for r in quality:
    cat = r.get("category", "unknown")
    cat_counts[cat] = cat_counts.get(cat, 0) + 1

print("\nCategory distribution:")
for cat, n in sorted(cat_counts.items()):
    bar = "█" * n
    print(f"  {cat:<25} {bar} ({n})")

# Average latency per category
print("\nAverage latency by category:")
cat_latency: dict[str, list] = {}
for r in quality:
    cat = r.get("category", "unknown")
    lat = r.get("metadata", {}).get("latency_ms", 0)
    cat_latency.setdefault(cat, []).append(lat)
for cat, lats in sorted(cat_latency.items()):
    avg = sum(lats) / len(lats)
    print(f"  {cat:<25} {avg:.1f} ms")
```

## Part 4: Skill extraction via LearningLoop

```python
# ── Skill extraction from passing sessions ───────────────────────────────────
# Live mode: call LearningLoop.maybe_write_skill(messages) for each passing session.
# Simulation: synthesise SKILL.md files from trajectory data.

SKILL_TEMPLATES: dict[str, tuple[str, str]] = {
    "arithmetic": (
        "arithmetic-calculation",
        "## Arithmetic calculation\n\nWhen asked for a numeric result, compute step by step "
        "and reply with only the final number (no units unless asked).\n"
        "Example: `1234 * 5678 = 7,006,652`\n",
    ),
    "code_generation": (
        "python-code-generation",
        "## Python code generation\n\nAlways include the `def` line and a docstring. "
        "Use type hints for function parameters.\n"
        "Example: `def add(a: int, b: int) -> int: ...`\n",
    ),
    "reasoning": (
        "step-by-step-reasoning",
        "## Step-by-step reasoning\n\nFor multi-step problems, show each calculation "
        "or logical step on its own line. Number enumerated items starting from 1.\n",
    ),
    "text_manipulation": (
        "text-manipulation",
        "## Text manipulation\n\nFor string operations, perform the operation mentally and "
        "respond with only the result unless the task asks for explanation.\n",
    ),
    "agent_knowledge": (
        "harness-agent-knowledge",
        "## Harness Agent knowledge\n\nKey facts:\n"
        "- Chapter 14: CronScheduler\n"
        "- Chapter 15: Gateway + CLI\n"
        "- SessionStore uses SQLite FTS5 for full-text search\n"
        "- SkillCatalog discovers SKILL.md files in labs/skills/\n",
    ),
}


def extract_skills_from_results(
    results: list[TaskResult],
    tasks: list[BenchmarkTask],
    skills_dir: Path,
    pass_rate_threshold: float = 0.5,
) -> int:
    """
    For categories where pass rate >= threshold, write a SKILL.md.
    Simulates LearningLoop.maybe_write_skill() per session.
    Returns the number of skills written.
    """
    task_map  = {t.task_id: t for t in tasks}
    cat_stats: dict[str, list[bool]] = {}
    for r in results:
        cat = task_map[r.task_id].category
        cat_stats.setdefault(cat, []).append(r.passed)

    # Live mode:
    # from harness_agent.learning.skill_writer import LearningLoop
    # loop = LearningLoop()
    # for r in [r for r in results if r.passed]:
    #     loop.maybe_write_skill(messages_for_session(r.session_id))

    written = 0
    for cat, passes in cat_stats.items():
        rate = sum(passes) / len(passes)
        if rate >= pass_rate_threshold and cat in SKILL_TEMPLATES:
            skill_name, skill_content = SKILL_TEMPLATES[cat]
            skill_dir = skills_dir / skill_name
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: {skill_name}\ndescription: Extracted from benchmark (pass_rate={rate:.0%})\n---\n\n"
                + skill_content
            )
            written += 1
            print(f"  Skill extracted: {skill_name}  (pass_rate={rate:.0%})")
    return written


skills_dir = Path(os.environ.get("HARNESS_AGENT_HOME", "labs")) / "skills"
print("=== Skill extraction from baseline results ===\n")
n_skills = extract_skills_from_results(baseline_results, BENCHMARK_TASKS, skills_dir)
print(f"\nSkills written: {n_skills}")
```

```python
# ── SkillCatalog: discover extracted skills ──────────────────────────────────

try:
    from harness_agent.skills.loader import SkillCatalog
    catalog = SkillCatalog(skills_dir=skills_dir)
    skills = catalog.discover()
    print(f"SkillCatalog discovered {len(skills)} skill(s):")
    for s in skills:
        desc = getattr(s, 'description', '(no description)')[:70]
        print(f"  - {s.name}: {desc}")
except ImportError:
    # Simulation: scan SKILL.md files
    skill_files = sorted(skills_dir.rglob("SKILL.md"))
    print(f"Skills in {skills_dir}: {len(skill_files)}")
    for sf in skill_files:
        # Read frontmatter name/description
        content = sf.read_text().splitlines()
        name_line = next((l for l in content if l.startswith("name:")), "")
        desc_line = next((l for l in content if l.startswith("description:")), "")
        name = name_line.replace("name:", "").strip() or sf.parent.name
        desc = desc_line.replace("description:", "").strip()[:60] or "(no description)"
        print(f"  - {name}: {desc}")
```

## Part 5: Skill injection and re-run

```python
# ── PromptBuilder: inject discovered skills into system prompt ───────────────

def build_skill_aware_system_prompt(skills_dir: Path) -> str:
    """
    Simulates PromptBuilder.build_system_prompt(skills_dir=skills_dir).
    In live mode: PromptBuilder reads SKILL.md files and injects them
    into the system prompt before the first turn.
    """
    skill_files = sorted(skills_dir.rglob("SKILL.md"))
    if not skill_files:
        return "You are Harness Agent."

    sections = ["You are Harness Agent.\n\n# Available Skills\n"]
    for sf in skill_files:
        content = sf.read_text()
        # Strip frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            content = parts[2].strip() if len(parts) >= 3 else content
        sections.append(content)

    return "\n\n---\n\n".join(sections)


# Live mode:
# from harness_agent.prompt.builder import PromptBuilder
# builder = PromptBuilder(skills_dir=skills_dir)
# system_prompt = builder.build_system_prompt()

system_prompt = build_skill_aware_system_prompt(skills_dir)
lines = system_prompt.splitlines()
print(f"System prompt with skills: {len(lines)} lines")
print()
# Show the first 20 lines
for line in lines[:20]:
    print(" ", line)
if len(lines) > 20:
    print(f"  ... ({len(lines) - 20} more lines)")
```

```python
# ── Re-run benchmark with skill-aware agent ──────────────────────────────────
# The skill-aware agent has the harness-agent-knowledge skill injected,
# which closes the knowledge gap for agent-01.

skilled_agent = MockAgent(skill_boost={"agent-01"})  # skill injection unlocks this task
skilled_results = run_benchmark(skilled_agent, BENCHMARK_TASKS)

# Comparison table
baseline_map = {r.task_id: r for r in baseline_results}
skilled_map  = {r.task_id: r for r in skilled_results}

print("=== Baseline vs. Skills-Injected comparison ===\n")
print(f"{'task_id':<15} {'base_pass':<12} {'skill_pass':<12} "
      f"{'lat_delta_ms':>14} {'tok_delta':>10}")
print("-" * 65)

improvements = 0
for task in BENCHMARK_TASKS:
    b = baseline_map[task.task_id]
    s = skilled_map[task.task_id]
    lat_delta = s.latency_ms - b.latency_ms
    tok_delta = (s.input_tokens + s.output_tokens) - (b.input_tokens + b.output_tokens)
    improved = (not b.passed and s.passed)
    if improved:
        improvements += 1
    marker = " ← improved" if improved else ""
    print(f"{task.task_id:<15} {'✓' if b.passed else '✗':<12} "
          f"{'✓' if s.passed else '✗':<12} "
          f"{lat_delta:>+14.1f} {tok_delta:>+10}{marker}")

print("-" * 65)
base_rate   = sum(r.passed for r in baseline_results) / len(baseline_results)
skilled_rate = sum(r.passed for r in skilled_results) / len(skilled_results)
print(f"\nPass rate:  baseline={base_rate:.0%}  with_skills={skilled_rate:.0%}  "
      f"delta=+{improvements} task(s)")
```

## Part 6: Fine-tuning pipeline setup

```python
# ── Fine-tuning pipeline: configs and conversion scripts ─────────────────────
# This cell shows how to connect the collected trajectories to three fine-tuning
# frameworks. No training is run here — this is the configuration layer.

LLAMAFACTORY_YAML = f"""
### LLaMA-Factory: fine-tune on benchmark trajectories
### Usage: llamafactory-cli train llama_factory_config.yaml

model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
stage: sft
do_train: true
finetuning_type: lora

dataset: harness_benchmark              # points to benchmark_trajectories.jsonl
dataset_dir: {Path('labs').resolve()}
template: llama3
cutoff_len: 2048

output_dir: labs/checkpoints/llama3-harness
num_train_epochs: 3
per_device_train_batch_size: 4
learning_rate: 2.0e-4
lora_rank: 16
lora_alpha: 32
"""

AXOLOTL_YAML = f"""
### Axolotl: fine-tune on benchmark trajectories
### Usage: accelerate launch -m axolotl.cli.train axolotl_config.yaml

base_model: mistralai/Mistral-7B-Instruct-v0.3
model_type: MistralForCausalLM

datasets:
  - path: {traj_path.resolve()}
    type: sharegpt
    conversation: chatml

sequence_len: 2048
output_dir: labs/checkpoints/mistral-harness
num_epochs: 3
micro_batch_size: 4
adapter: lora
lora_r: 16
"""

def convert_to_openai_format(jsonl_path: Path, out_path: Path) -> int:
    """
    Convert ShareGPT JSONL to OpenAI fine-tuning JSONL format.
    Usage: upload out_path to https://platform.openai.com/fine-tuning
    """
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    records = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w") as fh:
        for rec in records:
            messages = [
                {"role": role_map.get(m["from"], "user"), "content": m["value"]}
                for m in rec.get("conversations", [])
            ]
            fh.write(json.dumps({"messages": messages}) + "\n")
            count += 1
    return count


# Write config files and convert format
labs = Path(os.environ.get("HARNESS_AGENT_HOME", "labs"))
(labs / "llama_factory_config.yaml").write_text(LLAMAFACTORY_YAML)
(labs / "axolotl_config.yaml").write_text(AXOLOTL_YAML)

openai_path = labs / "benchmark_openai_format.jsonl"
n_converted = convert_to_openai_format(traj_path, openai_path)

print("Fine-tuning pipeline setup:")
print(f"  LLaMA-Factory config: {labs / 'llama_factory_config.yaml'}")
print(f"  Axolotl config:       {labs / 'axolotl_config.yaml'}")
print(f"  OpenAI format:        {openai_path}  ({n_converted} examples)")
print()
print("To fine-tune with OpenAI:")
print("  openai api fine_tuning.jobs.create \\")
print(f"    --training-file {openai_path} \\")
print("    --model gpt-4o-mini-2024-07-18")
```

## Part 7: Nightly automation — CronScheduler

```python
# ── Build and demo a nightly benchmark cron job ──────────────────────────────

def build_benchmark_cron_job() -> dict:
    """
    Returns a job payload compatible with CronScheduler (ch14).
    In live mode: write this to labs/cron/jobs.json and run
    `harness-agent cron tick` to execute nightly.
    """
    return {
        "id": "nightly-benchmark",
        "prompt": (
            "Run the self-improvement benchmark pipeline: "
            "1) execute benchmark suite, "
            "2) collect passing trajectories to labs/benchmark_trajectories.jsonl, "
            "3) extract skills to labs/skills/, "
            "4) re-run benchmark with skill injection, "
            "5) report pass rate delta."
        ),
        "schedule_minutes": 1440,   # 24 * 60 = nightly
        "last_run": None,
        "enabled": True,
    }


job = build_benchmark_cron_job()

# Persist to cron queue (for live mode)
cron_dir = Path(os.environ.get("HARNESS_AGENT_HOME", "labs")) / "cron"
cron_dir.mkdir(parents=True, exist_ok=True)
jobs_path = cron_dir / "jobs.json"

existing = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
# Upsert
updated = [j for j in existing if j.get("id") != job["id"]] + [job]
jobs_path.write_text(json.dumps(updated, indent=2))

print(f"Cron job registered: {jobs_path}")
print(json.dumps(job, indent=2))

# Simulate CronScheduler.tick()
try:
    from harness_agent.cron.scheduler import CronScheduler
    scheduler = CronScheduler()
    print("\nCronScheduler.tick() (simulation — no API key needed for dry-run):")
    # In live mode: scheduler.tick() dispatches to AIAgent
    jobs_loaded = scheduler.load_jobs()
    print(f"  Jobs in queue: {len(jobs_loaded)}")
    for j in jobs_loaded:
        print(f"  - {j.id}: every {j.schedule_minutes} min, last_run={j.last_run}")
except ImportError:
    print("\nCronScheduler not available — simulation only.")
    print(f"  Job '{job['id']}' registered for every {job['schedule_minutes']} minutes.")
    print("  Run `harness-agent cron tick` to execute nightly.")
```

## Part 8: Observability of improvement — BenchmarkMetrics

```python
import json as _json
from dataclasses import dataclass, field
from typing import Any

# ── Re-define MetricsCollector inline (origin: ch23) ─────────────────────────
class _Noop:
    def labels(self, **_): return self
    def observe(self, _):  pass
    def inc(self, _=1):    pass

class MetricsCollector:
    def __init__(self):
        try:
            from prometheus_client import Counter, Histogram, CollectorRegistry
            self._reg = CollectorRegistry()
            self.pass_rate    = Histogram("harness_bench_pass_rate",   "Pass rate per run",
                                          ["run"], buckets=[0.1*i for i in range(11)], registry=self._reg)
            self.avg_latency  = Histogram("harness_bench_latency_ms",  "Avg latency",
                                          ["run"], registry=self._reg)
            self.avg_tokens   = Histogram("harness_bench_tokens",      "Avg tokens",
                                          ["run"], registry=self._reg)
            self.skill_reuse  = Counter(  "harness_bench_skill_reuse", "Tasks helped by skills",
                                          ["run"], registry=self._reg)
        except ImportError:
            self.pass_rate = self.avg_latency = self.avg_tokens = self.skill_reuse = _Noop()


@dataclass
class BenchmarkMetrics:
    """
    Tracks benchmark KPIs across multiple runs.
    Renders a text-based trend table for observability.
    """
    history: list[dict] = field(default_factory=list)
    _collector: MetricsCollector = field(default_factory=MetricsCollector, init=False)

    def record_run(
        self,
        run_label: str,
        results: list[TaskResult],
        skill_improved: int = 0,
    ) -> None:
        total = len(results)
        passed = sum(r.passed for r in results)
        pass_rate = passed / total
        avg_lat = sum(r.latency_ms for r in results) / total
        avg_tok = sum(r.input_tokens + r.output_tokens for r in results) / total
        skill_reuse_rate = skill_improved / total

        self.history.append({
            "run": run_label,
            "pass_rate": pass_rate,
            "avg_latency_ms": avg_lat,
            "avg_tokens": avg_tok,
            "skill_reuse_rate": skill_reuse_rate,
            "passed": passed,
            "total": total,
        })
        self._collector.pass_rate.labels(run=run_label).observe(pass_rate)
        self._collector.avg_latency.labels(run=run_label).observe(avg_lat)
        self._collector.avg_tokens.labels(run=run_label).observe(avg_tok)
        self._collector.skill_reuse.labels(run=run_label).inc(skill_improved)

    def render_trend(self) -> None:
        if not self.history:
            print("No runs recorded yet.")
            return
        print(f"{'Run':<20} {'Pass%':>7} {'AvgLat(ms)':>12} {'AvgTok':>8} {'SkillReuse%':>12}")
        print("-" * 65)
        for h in self.history:
            bar_width = int(h["pass_rate"] * 20)
            bar = "█" * bar_width + "░" * (20 - bar_width)
            print(
                f"{h['run']:<20} {h['pass_rate']:>6.0%} "
                f"{h['avg_latency_ms']:>12.1f} "
                f"{h['avg_tokens']:>8.0f} "
                f"{h['skill_reuse_rate']:>11.0%}"
            )
            print(f"  [{bar}] {h['passed']}/{h['total']} tasks")


# Record both runs
bench_metrics = BenchmarkMetrics()
bench_metrics.record_run("baseline",      baseline_results, skill_improved=0)
bench_metrics.record_run("with_skills",   skilled_results,  skill_improved=improvements)

# Simulate a third run (hypothetical future improvement)
future_agent = MockAgent(skill_boost={"agent-01"})  # same improvement held
future_results = run_benchmark(future_agent, BENCHMARK_TASKS)
bench_metrics.record_run("run_3_nightly", future_results, skill_improved=improvements)

print("=== BenchmarkMetrics trend ===\n")
bench_metrics.render_trend()
```

## The flywheel in full — retrospective

```
Run 0 (baseline)
  └── 9/10 pass  →  9 trajectories collected
                 →  4 skills extracted (arithmetic, code, reasoning, text)
                 →  harness-agent-knowledge skill: agent-01 now passing
Run 1 (with skills)
  └── 10/10 pass  →  10 trajectories collected (all quality examples)
                  →  skills reinforced; latency ↓ (fewer reasoning hops)
                  →  tokens ↓ (shorter prompts for familiar tasks)
Run 2 (nightly, automated)
  └── 10/10 pass  →  fine-tuning dataset grows
                  →  skill_reuse_rate ↑ (agent matches patterns faster)
                  →  CronScheduler dispatches pipeline automatically

Each cycle: more data → better skills → higher pass rate → cheaper inference.
The agent learns from its own successes.
```

### Chapter cross-reference — what each chapter contributed

| Capability | Key class / call | Chapter |
|---|---|---|
| Agent loop | `AIAgent.run_conversation()` | ch03 |
| Tool registry | `get_registry()` | ch04 |
| Session store | `SessionStore` | ch06 |
| Prompt assembly + skill injection | `PromptBuilder` | ch07 |
| Skill catalog | `SkillCatalog.discover()` | ch08 |
| Learning loop | `LearningLoop.maybe_write_skill()` | ch10 |
| Cron automation | `CronScheduler.tick()` | ch14 |
| Trajectory export | `export_trajectories()` | ch19 |
| Multi-agent investigation | `MultiAgentOrchestrator` | ch22 |
| Observability | `MetricsCollector`, `instrument_turn` | ch23 |
| Reliability | `CircuitBreaker`, `ProviderFallbackChain` | ch25 |

## Hands-on exercises

1. **Live benchmark**: Set `ANTHROPIC_API_KEY`, replace `MockAgent` with `AIAgent()`, and run the benchmark against the real model. Compare your pass rate to the simulation baseline.

2. **Expand the benchmark**: Add 5 more tasks to `BENCHMARK_TASKS` — one for each of: JSON parsing, regex writing, multi-file reasoning, error debugging, and prompt rewriting. Verify `len(BENCHMARK_TASKS) == 15` still satisfies the runner.

3. **Adaptive skill threshold**: Modify `extract_skills_from_results()` to also write a skill when a task fails (to capture the "what not to do" pattern). Add a `negative_example: true` frontmatter field to these skills.

4. **Fine-tuning round-trip**: Run the LLaMA-Factory config against a locally quantised model using `ollama` or `llama.cpp`. Benchmark the fine-tuned checkpoint — does pass rate improve?

5. **Multi-run convergence**: Run the benchmark 5 times (with skills injected from each prior run). Plot `pass_rate`, `avg_latency_ms`, and `avg_tokens` across runs using `matplotlib`. Observe the convergence.

6. **Skill decay detection**: After 10 nightly runs, add a task whose `success_fn` always returns `False` (unsolvable). Verify `BenchmarkMetrics` captures the regression and the `CronScheduler` DLQ logs the failure.

7. **Cross-capstone flywheel**: Feed incident investigation transcripts from ch26 (the `labs/incidents.jsonl`) into this benchmark's `extract_skills_from_results()`. Verify SRE runbooks appear in `labs/skills/`.

8. **OpenAI fine-tuning upload**: Upload `benchmark_openai_format.jsonl` to the OpenAI fine-tuning API. Use the resulting model ID as a provider in `ProviderRegistry` (ch16) and benchmark it against the base model.

## Common pitfalls

| Pitfall | Symptom | Diagnosis |
|---|---|---|
| `success_fn` too strict | Correct answer fails (e.g. `"7006652"` not matched due to comma) | Normalise response before checking: `.replace(",", "").replace(" ", "")` |
| Extracting skills from failing tasks | `LearningLoop` writes a skill for an incorrect pattern | Only extract from `results where r.passed == True`; gate on `pass_rate_threshold` |
| Skill injection bloats the context window | System prompt grows past 12k tokens | Cap the number of injected skills; use `compress_messages()` (ch11) if needed |
| Benchmark sessions not stored | `export_trajectories()` returns 0 | `AIAgent` must store sessions in `SessionStore`; verify `HARNESS_AGENT_HOME` is set |
| Cron job fires before prior run finishes | Two pipeline runs overlap; DLQ fills | Add an in-flight lock (file or DB flag) to `build_benchmark_cron_job` |
| Fine-tuning format mismatch | LLaMA-Factory rejects JSONL | Verify `type: sharegpt` and `conversation: chatml` match the exported format |
| `BenchmarkMetrics` not persisted across sessions | Trend resets on every notebook restart | Write `history` to `labs/benchmark_history.json` and reload at startup |

## Checkpoint questions

1. Describe the five stages of the self-improvement flywheel. What is the input and output of each stage?
2. Why does `extract_skills_from_results()` use `pass_rate_threshold=0.5` rather than extracting from every passing task individually?
3. `PromptBuilder` injects skills into the system prompt. What happens to latency and token count when too many skills are injected? How does ch11 help?
4. What is the difference between the `collect_trajectories()` simulation path and the live `export_trajectories()` call? Which subsystem stores the data in the live path?
5. A task passes in the baseline but fails after skill injection. What are three possible causes?
6. Why must `BenchmarkTask.success_fn` be a pure function with no side effects?
7. Explain how `CronScheduler` + `build_benchmark_cron_job()` enables the flywheel to run autonomously without human intervention.

## Final summary

| Concept | Key detail |
|---|---|
| `BenchmarkTask` | `task_id` + `prompt` + `success_fn(response) → bool`; 10 tasks covering arithmetic, code, reasoning, text, agent knowledge |
| `run_benchmark()` | Runs all tasks; returns `list[TaskResult]` with `passed`, `latency_ms`, `input_tokens`, `output_tokens`, `session_id` |
| `collect_trajectories()` | Exports passing sessions to ShareGPT JSONL via `export_trajectories()`; simulation path synthesises records |
| `extract_skills_from_results()` | Writes `SKILL.md` for categories where `pass_rate >= threshold`; simulates `LearningLoop.maybe_write_skill()` |
| `build_skill_aware_system_prompt()` | Simulates `PromptBuilder` reading `labs/skills/**/*.md` and injecting into the system prompt |
| `convert_to_openai_format()` | Converts ShareGPT JSONL to OpenAI fine-tuning format (role: system/user/assistant) |
| `build_benchmark_cron_job()` | Returns `CronScheduler`-compatible job payload; schedules pipeline every 1440 min (nightly) |
| `BenchmarkMetrics` | Records `pass_rate`, `avg_latency_ms`, `avg_tokens`, `skill_reuse_rate` per run; renders ASCII trend table |
| Self-improvement flywheel | Each cycle: run → collect → extract → inject → measure; pass rate ↑, latency ↓, tokens ↓ |

---

### Capstone B certification checklist

- [ ] `BENCHMARK_TASKS` contains exactly 10 tasks across 5 categories
- [ ] `run_benchmark(mock_agent, BENCHMARK_TASKS)` returns `list[TaskResult]` with 10 entries
- [ ] `collect_trajectories()` writes at least one JSONL record to `labs/benchmark_trajectories.jsonl`
- [ ] `extract_skills_from_results()` writes at least one `SKILL.md` to `labs/skills/`
- [ ] `build_skill_aware_system_prompt()` returns a string containing skill content
- [ ] Skills-injected run shows higher pass rate than baseline for at least one task
- [ ] `convert_to_openai_format()` writes a valid `benchmark_openai_format.jsonl`
- [ ] `build_benchmark_cron_job()` registers a job in `labs/cron/jobs.json`
- [ ] `BenchmarkMetrics.render_trend()` shows at least 2 runs with pass rate and latency

---

### Full tutorial certification checklist (ch00–ch27)

- [ ] **ch00–ch02**: Environment setup; first API call; provider abstraction
- [ ] **ch03–ch05**: Agent loop; tool registry; observations and recovery
- [ ] **ch06–ch09**: Session storage; prompt assembly; skills; memory
- [ ] **ch10–ch13**: Learning loop; context compression; subagents; MCP integration
- [ ] **ch14–ch17**: Cron scheduler; gateway + CLI; provider resolution; terminal backends
- [ ] **ch18–ch21**: ACP integration; trajectories; plugins; full system integration
- [ ] **ch22**: Multi-agent orchestration (GraphAgent, MultiAgentOrchestrator)
- [ ] **ch23**: Production observability (logs, traces, metrics, health probes)
- [ ] **ch24**: Scalable multi-tenant architecture (TenantContext, RateLimiter, SessionRouter)
- [ ] **ch25**: Reliability and resilience (CircuitBreaker, retry, fallback chain, DLQ)
- [ ] **ch26**: Capstone A — incident response bot wiring all 25 chapters
- [ ] **ch27**: Capstone B — self-improving agent flywheel (benchmark → collect → extract → inject → measure)

**Congratulations — you have built, deployed, and improved Harness Agent from first principles.**

The flywheel is now running. Every successful agent interaction makes the next one cheaper and better.
