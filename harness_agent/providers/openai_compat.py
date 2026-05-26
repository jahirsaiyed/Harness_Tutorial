"""OpenAI-compatible chat completions provider."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from harness_agent.providers.base import BaseProvider, resolve_api_key
from harness_agent.types import Message, ToolCall


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key or resolve_api_key("OPENAI_API_KEY"),
                base_url=self._base_url,
            )
        return self._client

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        model: str,
    ) -> tuple[str | None, list[ToolCall]]:
        response = self._get_client().chat.completions.create(
            model=model,
            messages=[m.to_openai() for m in messages],
            tools=tools or None,
        )
        choice = response.choices[0].message
        text = choice.content
        calls: list[ToolCall] = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )
        return text, calls
