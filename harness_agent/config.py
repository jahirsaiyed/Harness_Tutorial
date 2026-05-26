"""Configuration and paths for Harness Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _home() -> Path:
    raw = os.environ.get("HARNESS_AGENT_HOME", "./labs")
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class HarnessConfig:
    home: Path = field(default_factory=_home)
    default_provider: str = field(
        default_factory=lambda: os.environ.get("HARNESS_DEFAULT_PROVIDER", "openai")
    )
    default_model: str = field(
        default_factory=lambda: os.environ.get("HARNESS_DEFAULT_MODEL", "gpt-4o-mini")
    )
    max_turns: int = 25
    terminal_backend: str = "local"  # local | docker
    reference_repo_path: Path | None = None

    def __post_init__(self) -> None:
        ref = os.environ.get("REFERENCE_REPO_PATH", "").strip()
        if ref:
            self.reference_repo_path = Path(ref).expanduser().resolve()
        for sub in ("workspace", "skills", "memory", "cron", "sessions"):
            (self.home / sub).mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.home / "sessions" / "harness_agent.db"

    @property
    def soul_path(self) -> Path:
        return self.home / "SOUL.md"

    @property
    def memory_path(self) -> Path:
        return self.home / "memory" / "MEMORY.md"

    @property
    def user_path(self) -> Path:
        return self.home / "memory" / "USER.md"

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def cron_jobs_path(self) -> Path:
        return self.home / "cron" / "jobs.json"


def get_config() -> HarnessConfig:
    return HarnessConfig()
