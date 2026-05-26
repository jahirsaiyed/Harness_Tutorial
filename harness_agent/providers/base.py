"""Provider abstraction for LLM completion with tools."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from harness_agent.types import Message, ToolCall


def resolve_api_key(*provider_env_vars: str) -> str | None:
    """Return the first set env var, falling back to HARNESS_API_KEY.

    Resolution order:
    1. Provider-specific env vars (e.g. OPENAI_API_KEY, COMPASS_API_KEY)
    2. Universal fallback: HARNESS_API_KEY
    """
    for var in provider_env_vars:
        value = os.environ.get(var)
        if value:
            return value
    return os.environ.get("HARNESS_API_KEY")


class BaseProvider(ABC):
    name: str

    @abstractmethod
    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        model: str,
    ) -> tuple[str | None, list[ToolCall]]:
        """Return assistant text (if any) and tool calls."""


def complete_with_tools(
    provider: BaseProvider,
    messages: list[Message],
    tools: list[dict[str, Any]],
    *,
    model: str,
) -> tuple[str | None, list[ToolCall]]:
    return provider.complete_with_tools(messages, tools, model=model)
