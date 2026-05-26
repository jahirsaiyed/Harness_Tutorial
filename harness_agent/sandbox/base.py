"""Sandbox execution backend — unified interface for safe code/command execution.

Three backends are provided:

  local   — subprocess in workspace dir (dev/testing, no isolation)
  docker  — ephemeral container per command, workspace mounted (strong isolation)
  e2b     — cloud microVM via e2b.dev (strongest isolation, requires API key)

Select via HARNESS_SANDBOX env var or the terminal_backend config field.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class SandboxBackend(ABC):
    """Uniform interface for sandboxed command and file operations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'local', 'docker', or 'e2b'."""

    @abstractmethod
    def run(self, command: str, *, timeout: int = 60) -> tuple[int, str, str]:
        """Execute a shell command. Returns (exit_code, stdout, stderr)."""

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Write content to a file inside the sandbox at the given relative path."""

    @abstractmethod
    def read_file(self, path: str) -> str:
        """Read a file from inside the sandbox. Raises FileNotFoundError if missing."""


def get_sandbox() -> SandboxBackend:
    """Factory: resolve sandbox backend from HARNESS_SANDBOX env var or config.

    Resolution order:
      1. HARNESS_SANDBOX environment variable ('local', 'docker', 'e2b')
      2. HarnessConfig.terminal_backend ('local' | 'docker')
      3. Default: 'local'
    """
    backend_name = os.environ.get("HARNESS_SANDBOX", "").strip()
    if not backend_name:
        from harness_agent.config import get_config
        backend_name = get_config().terminal_backend

    if backend_name == "docker":
        from harness_agent.sandbox.docker_sandbox import DockerSandbox
        return DockerSandbox()
    if backend_name == "e2b":
        from harness_agent.sandbox.e2b_sandbox import E2BSandbox
        return E2BSandbox()
    from harness_agent.sandbox.local_sandbox import LocalSandbox
    return LocalSandbox()
