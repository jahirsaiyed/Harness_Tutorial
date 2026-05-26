"""Structured tool observations for Harness Agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

Status = Literal["success", "warning", "error"]


@dataclass
class Observation:
    status: Status
    summary: str
    next_actions: list[str]
    artifacts: list[str]
    detail: str | None = None

    def to_model_content(self) -> str:
        payload = {
            "status": self.status,
            "summary": self.summary,
            "next_actions": self.next_actions,
            "artifacts": self.artifacts,
        }
        if self.detail:
            payload["detail"] = self.detail
        return json.dumps(payload, indent=2)


def wrap_result(
    *,
    status: Status,
    summary: str,
    next_actions: list[str] | None = None,
    artifacts: list[str] | None = None,
    detail: str | None = None,
) -> str:
    obs = Observation(
        status=status,
        summary=summary,
        next_actions=next_actions or [],
        artifacts=artifacts or [],
        detail=detail,
    )
    return obs.to_model_content()


def wrap_exception(exc: BaseException, *, retry_hint: str | None = None) -> str:
    actions = ["Fix the inputs and retry once."]
    if retry_hint:
        actions.insert(0, retry_hint)
    return wrap_result(
        status="error",
        summary=str(exc),
        next_actions=actions,
        detail=type(exc).__name__,
    )
