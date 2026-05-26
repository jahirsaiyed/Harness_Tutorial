"""Provider abstraction for LLM completion with tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from harness_agent.types import Message, ToolCall


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
