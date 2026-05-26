"""Local subprocess terminal backend."""

from __future__ import annotations

import subprocess

from harness_agent.config import get_config
from harness_agent.tools.environments.base import TerminalBackend


class LocalBackend(TerminalBackend):
    def run(self, command: str, *, cwd: str | None = None, timeout: int = 60) -> tuple[int, str, str]:
        work = cwd or str(get_config().home / "workspace")
        proc = subprocess.run(
            command,
            shell=True,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
