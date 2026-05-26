"""System prompt assembly for Harness Agent."""

from __future__ import annotations

from pathlib import Path

from harness_agent.config import get_config
from harness_agent.memory.files import load_memory_block
from harness_agent.skills.loader import SkillCatalog


class PromptBuilder:
    def __init__(self) -> None:
        self.config = get_config()
        self.skills = SkillCatalog(self.config.skills_dir)

    def _read_optional(self, path: Path) -> str:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        return ""

    def build_system_prompt(self, *, workspace_context: str | None = None) -> str:
        parts: list[str] = [
            "You are Harness Agent, a capable assistant with tools.",
            "Follow tool JSON observations. Prefer concise, correct actions.",
        ]
        soul = self._read_optional(self.config.soul_path)
        if soul:
            parts.append(f"# Persona (SOUL)\n{soul}")
        memory_block = load_memory_block()
        if memory_block:
            parts.append(memory_block)
        skill_meta = self.skills.metadata_block()
        if skill_meta:
            parts.append(skill_meta)
        harness_ctx = self.config.home / "workspace" / ".harness.md"
        ctx = self._read_optional(harness_ctx)
        if ctx:
            parts.append(f"# Project context\n{ctx}")
        if workspace_context:
            parts.append(f"# Workspace note\n{workspace_context}")
        return "\n\n".join(parts)
