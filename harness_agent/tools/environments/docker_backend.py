"""Docker terminal backend (optional — requires Docker)."""

from __future__ import annotations

import shutil
import subprocess

from harness_agent.config import get_config
from harness_agent.tools.environments.base import TerminalBackend


def docker_available() -> bool:
    return shutil.which("docker") is not None


class DockerBackend(TerminalBackend):
    def run(self, command: str, *, cwd: str | None = None, timeout: int = 60) -> tuple[int, str, str]:
        work = cwd or str(get_config().home / "workspace")
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{work}:/workspace",
                "-w",
                "/workspace",
                "python:3.11-slim",
                "bash",
                "-lc",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
