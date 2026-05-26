"""Terminal backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness_agent.config import get_config


class TerminalBackend(ABC):
    @abstractmethod
    def run(self, command: str, *, cwd: str | None = None, timeout: int = 60) -> tuple[int, str, str]:
        """Return exit code, stdout, stderr."""


def get_terminal_backend() -> TerminalBackend:
    from harness_agent.tools.environments.docker_backend import DockerBackend
    from harness_agent.tools.environments.local import LocalBackend

    name = get_config().terminal_backend
    if name == "docker":
        return DockerBackend()
    return LocalBackend()
