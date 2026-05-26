"""Closed learning loop — post-turn skill authoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from harness_agent.config import get_config


@dataclass
class LearningLoop:
    tool_call_threshold: int = 5

    def should_author_skill(self, tool_call_count: int, had_error_recovery: bool = False) -> bool:
        if tool_call_count >= self.tool_call_threshold:
            return True
        return had_error_recovery and tool_call_count >= 2

    def nudge_message(self) -> str:
        return (
            "[Harness Agent learning nudge] If this task produced a reusable workflow, "
            "consider persisting it as a skill for future sessions."
        )


def maybe_write_skill(
    *,
    skill_name: str,
    description: str,
    body: str,
    tool_call_count: int,
) -> Path | None:
    loop = LearningLoop()
    if not loop.should_author_skill(tool_call_count):
        return None
    cfg = get_config()
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", skill_name).strip("-").lower() or "workflow"
    skill_dir = cfg.skills_dir / safe
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    stamp = datetime.now(timezone.utc).isoformat()
    content = f"""---
name: {safe}
description: {description}
created: {stamp}
---

# {skill_name}

{body}
"""
    path.write_text(content, encoding="utf-8")
    return path
