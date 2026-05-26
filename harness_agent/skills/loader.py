"""Skill catalog with progressive disclosure."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from harness_agent.observations import wrap_result
from harness_agent.tools.registry import register


@dataclass
class SkillMeta:
    name: str
    description: str
    path: Path


class SkillCatalog:
    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def discover(self) -> list[SkillMeta]:
        metas: list[SkillMeta] = []
        for path in sorted(self.skills_dir.glob("**/SKILL.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            name = path.parent.name
            desc = ""
            m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
            if m:
                desc = m.group(1).strip()
            elif text.startswith("#"):
                desc = text.split("\n", 1)[0].lstrip("# ").strip()
            metas.append(SkillMeta(name=name, description=desc[:200], path=path))
        return metas

    def metadata_block(self) -> str:
        lines = ["# Available skills (metadata only — request full skill when needed)"]
        for meta in self.discover():
            lines.append(f"- **{meta.name}**: {meta.description or '(no description)'}")
        return "\n".join(lines)

    def load_full(self, name: str) -> str | None:
        for meta in self.discover():
            if meta.name == name:
                return meta.path.read_text(encoding="utf-8")
        return None


def load_skill(name: str) -> str:
    catalog = SkillCatalog(Path(__file__).resolve().parents[2] / ".." / "labs" / "skills")
    from harness_agent.config import get_config

    catalog = SkillCatalog(get_config().skills_dir)
    body = catalog.load_full(name)
    if not body:
        return wrap_result(status="error", summary=f"Unknown skill: {name}", next_actions=["List skills metadata."])
    return wrap_result(status="success", summary=f"Loaded skill {name}", detail=body[:12000])


register(
    name="load_skill",
    description="Load full SKILL.md body for a named skill (progressive disclosure level 2).",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    handler=load_skill,
    toolset="skills",
)
