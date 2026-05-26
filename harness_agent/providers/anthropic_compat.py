"""Anthropic messages API provider."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from harness_agent.providers.base import BaseProvider, resolve_api_key
from harness_agent.types import Message, ToolCall


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self._api_key or resolve_api_key("ANTHROPIC_API_KEY"),
            )
        return self._client

    def _to_anthropic(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                if m.content:
                    system_parts.append(m.content)
                continue
            if m.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content or "",
                            }
                        ],
                    }
                )
                continue
            if m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"])
                            if isinstance(tc["function"]["arguments"], str)
                            else tc["function"]["arguments"],
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
                continue
            out.append({"role": m.role, "content": m.content or ""})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, out

    def _tools_to_anthropic(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for t in tools:
            fn = t.get("function", {})
            result.append(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return result

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        model: str,
    ) -> tuple[str | None, list[ToolCall]]:
        system, msgs = self._to_anthropic(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": msgs,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._tools_to_anthropic(tools)
        response = self._get_client().messages.create(**kwargs)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input),
                    )
                )
        text = "\n".join(text_parts) if text_parts else None
        return text, calls
