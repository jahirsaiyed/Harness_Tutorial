"""Local subprocess sandbox — no container isolation, workspace-scoped paths."""

from __future__ import annotations

import subprocess
from pathlib import Path

from harness_agent.config import get_config
from harness_agent.sandbox.base import SandboxBackend


class LocalSandbox(SandboxBackend):
    """Runs commands via subprocess in the agent workspace directory.

    No container isolation — use Docker or e2b for untrusted code.
    Suitable for development, testing, and trusted workflows.
    """

    @property
    def name(self) -> str:
        return "local"

    def _workspace(self) -> Path:
        p = get_config().home / "workspace"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def run(self, command: str, *, timeout: int = 60) -> tuple[int, str, str]:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=self._workspace(),
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
