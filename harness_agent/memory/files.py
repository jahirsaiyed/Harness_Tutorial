"""MEMORY.md and USER.md injection."""

from __future__ import annotations

from harness_agent.config import get_config


def load_memory_block() -> str:
    cfg = get_config()
    parts: list[str] = []
    if cfg.memory_path.is_file():
        parts.append(f"# Long-term memory (MEMORY)\n{cfg.memory_path.read_text(encoding='utf-8').strip()}")
    if cfg.user_path.is_file():
        parts.append(f"# User model (USER)\n{cfg.user_path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(parts)
