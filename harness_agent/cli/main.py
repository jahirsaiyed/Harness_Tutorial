"""Typer CLI for Harness Agent."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from harness_agent.agent import AIAgent
from harness_agent.config import get_config
from harness_agent.cron.scheduler import CronScheduler
from harness_agent.gateway.runner import GatewayRunner
from harness_agent.plugins.loader import discover_plugins
from harness_agent.providers.registry import get_provider_registry
from harness_agent.trajectories.export import export_trajectories

app = typer.Typer(name="harness-agent", help="Harness Agent — tutorial CLI")
console = Console()


@app.command()
def doctor() -> None:
    """Check environment, paths, and provider keys."""
    cfg = get_config()
    issues: list[str] = []
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        issues.append("No OPENAI_API_KEY or ANTHROPIC_API_KEY set")
    if issues:
        console.print(Panel("\n".join(issues), title="Issues", style="red"))
        raise typer.Exit(1)
    console.print(Panel(f"HARNESS_AGENT_HOME={cfg.home}\nProviders: openai, anthropic", title="OK", style="green"))


@app.command()
def chat(
    message: str | None = typer.Argument(None, help="Optional one-shot message"),
    provider: str | None = typer.Option(None, "--provider", "-p"),
    model: str | None = typer.Option(None, "--model", "-m"),
) -> None:
    """Interactive chat or single message."""
    discover_plugins()
    agent = AIAgent()
    session_id: str | None = None
    if message:
        result = agent.run_conversation(message, session_id=session_id, provider=provider, model=model)
        console.print(result.assistant_text)
        return
    console.print("Harness Agent chat (Ctrl+C to exit)")
    while True:
        try:
            user = console.input("[bold cyan]you>[/] ")
        except (EOFError, KeyboardInterrupt):
            break
        if not user.strip():
            continue
        if user.strip().startswith("/model"):
            parts = user.split()
            if len(parts) >= 3:
                provider, model = parts[1], parts[2]
                console.print(f"Using {provider}:{model}")
            continue
        result = agent.run_conversation(user, session_id=session_id, provider=provider, model=model)
        session_id = result.session_id
        console.print(f"[bold green]agent>[/] {result.assistant_text}")


gateway_app = typer.Typer(help="Gateway commands")
app.add_typer(gateway_app, name="gateway")


@gateway_app.command("run")
def gateway_run() -> None:
    GatewayRunner().run_http()


cron_app = typer.Typer(help="Cron commands")
app.add_typer(cron_app, name="cron")


@cron_app.command("tick")
def cron_tick() -> None:
    results = CronScheduler().tick()
    console.print(results)


@app.command()
def acp() -> None:
    from harness_agent.acp.server import run_acp_stdio

    run_acp_stdio()


@app.command("export-trajectories")
def export_traj(
    output: Path = typer.Option(Path("labs/trajectories.jsonl"), "--output", "-o"),
) -> None:
    n = export_trajectories(output)
    console.print(f"Exported {n} sessions to {output}")


if __name__ == "__main__":
    app()
