"""File tools — read, write, search within workspace."""

from __future__ import annotations

from pathlib import Path

from harness_agent.config import get_config
from harness_agent.observations import wrap_result
from harness_agent.tools.registry import register


def _workspace() -> Path:
    return get_config().home / "workspace"


def _safe_path(rel: str) -> Path:
    base = _workspace().resolve()
    target = (base / rel).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError("Path escapes workspace")
    return target


def read_file(path: str) -> str:
    p = _safe_path(path)
    if not p.is_file():
        return wrap_result(status="error", summary=f"Not a file: {path}", next_actions=["Check path."])
    content = p.read_text(encoding="utf-8", errors="replace")
    return wrap_result(
        status="success",
        summary=f"Read {len(content)} chars from {path}",
        artifacts=[str(p)],
        detail=content[:8000],
    )


def write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return wrap_result(status="success", summary=f"Wrote {path}", artifacts=[str(p)])


def search_files(pattern: str) -> str:
    base = _workspace()
    matches = [str(p.relative_to(base)) for p in base.rglob(pattern) if p.is_file()][:50]
    return wrap_result(
        status="success",
        summary=f"Found {len(matches)} files",
        artifacts=matches,
    )


def _register() -> None:
    register(
        name="read_file",
        description="Read a text file from the agent workspace.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path under workspace"}},
            "required": ["path"],
        },
        handler=read_file,
        toolset="files",
    )
    register(
        name="write_file",
        description="Write text to a file in the agent workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        handler=write_file,
        toolset="files",
    )
    register(
        name="search_files",
        description="Glob search files under workspace.",
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string", "description": "Glob like *.py"}},
            "required": ["pattern"],
        },
        handler=search_files,
        toolset="files",
    )


_register()
