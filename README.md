# Harness Agent From Scratch

A chapter-by-chapter tutorial for building **Harness Agent** — a complete, self-improving agent harness — with Jupyter notebooks (`ch00`–`ch21`) and the `harness_agent` Python package.

## What you build

- LLM tool-calling loop with provider abstraction  
- Tool registry, observations, sessions (SQLite + FTS5)  
- Prompt assembly, skills, memory, learning loop, compression  
- Subagents, MCP, cron, gateway, CLI, ACP, trajectories, plugins  

## Quick start

```bash
cd Harness_Tutorial
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"
copy .env.example .env   # set OPENAI_API_KEY or ANTHROPIC_API_KEY
set HARNESS_AGENT_HOME=./labs
harness-agent doctor
harness-agent chat
```

## CLI

| Command | Description |
|---------|-------------|
| `harness-agent doctor` | Environment check |
| `harness-agent chat` | Interactive REPL |
| `harness-agent gateway run` | HTTP webhook gateway |
| `harness-agent cron tick` | Run due cron jobs |
| `harness-agent acp` | stdio JSON-RPC for IDE integration |
| `harness-agent export-trajectories` | Export JSONL |

## Notebooks

Open `notebooks/` in order: `ch00_introduction.ipynb` through `ch21_full_system_integration.ipynb`.

Optional: clone an external reference for comparison reading:

```bash
git clone https://github.com/NousResearch/hermes-agent.git ../reference-agent
# REFERENCE_REPO_PATH=../reference-agent in .env
```

## Docs

- [SYLLABUS.md](docs/SYLLABUS.md)  
- [GLOSSARY.md](docs/GLOSSARY.md)  
- [NOTEBOOK_AUTHORING.md](docs/NOTEBOOK_AUTHORING.md)  

## License

MIT — tutorial code for learning.
