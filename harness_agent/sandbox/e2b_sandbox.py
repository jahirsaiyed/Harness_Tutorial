"""e2b cloud sandbox — fully isolated ephemeral microVM via e2b.dev.

Each E2BSandbox instance owns one Sandbox VM that is killed when the
object is garbage-collected or __del__ is called.

Requirements:
  pip install e2b-code-interpreter
  E2B_API_KEY=<your key>   (https://e2b.dev/dashboard)

The e2b sandbox gives the strongest isolation guarantees:
  - Runs in a Firecracker microVM on e2b infrastructure
  - Full internet access available by default (can be restricted)
  - Persistent filesystem within a session
  - Suitable for untrusted or user-submitted code

Usage:
  sandbox = E2BSandbox()
  code, out, err = sandbox.run("python3 -c 'print(42)'")
  sandbox.write_file("script.py", "print('hello')")
  code, out, err = sandbox.run("python3 script.py")
"""

from __future__ import annotations

from harness_agent.sandbox.base import SandboxBackend


class E2BSandbox(SandboxBackend):
    def __init__(self) -> None:
        try:
            from e2b_code_interpreter import Sandbox  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "e2b sandbox requires: pip install e2b-code-interpreter\n"
                "Get an API key at https://e2b.dev/dashboard and set E2B_API_KEY."
            ) from exc
        self._sbx = Sandbox()

    @property
    def name(self) -> str:
        return "e2b"

    def run(self, command: str, *, timeout: int = 60) -> tuple[int, str, str]:
        result = self._sbx.commands.run(command, timeout=timeout)
        exit_code = result.exit_code if result.exit_code is not None else 0
        return exit_code, result.stdout or "", result.stderr or ""

    def write_file(self, path: str, content: str) -> None:
        self._sbx.files.write(path, content.encode())

    def read_file(self, path: str) -> str:
        return self._sbx.files.read(path).decode()

    def __del__(self) -> None:
        try:
            self._sbx.kill()
        except Exception:  # noqa: BLE001
            pass
