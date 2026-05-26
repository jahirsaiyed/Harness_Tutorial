"""Docker sandbox — ephemeral throwaway container per command execution.

Each run() call spins up a fresh python:3.11-slim container with:
  - No network access (--network none)
  - Memory cap (--memory 512m)
  - CPU cap (--cpus 1)
  - Workspace mounted at /workspace

Files written via write_file() persist in the host workspace directory
and are visible to subsequent run() calls via the volume mount.

Requires Docker to be running. Falls back gracefully if unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from harness_agent.config import get_config
from harness_agent.sandbox.base import SandboxBackend


def docker_available() -> bool:
    return shutil.which("docker") is not None


class DockerSandbox(SandboxBackend):
    IMAGE = "python:3.11-slim"

    @property
    def name(self) -> str:
        return "docker"

    def _workspace(self) -> Path:
        p = get_config().home / "workspace"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def run(self, command: str, *, timeout: int = 60) -> tuple[int, str, str]:
        work = str(self._workspace())
        proc = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", "512m",
                "--cpus", "1",
                "-v", f"{work}:/workspace",
                "-w", "/workspace",
                self.IMAGE,
                "bash", "-lc", command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def write_file(self, path: str, content: str) -> None:
        p = self._workspace() / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def read_file(self, path: str) -> str:
        p = self._workspace() / path
        if not p.is_file():
            raise FileNotFoundError(path)
        return p.read_text(encoding="utf-8")
