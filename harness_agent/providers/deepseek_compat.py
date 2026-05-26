"""DeepSeek provider — OpenAI-compatible API at api.deepseek.com."""

from __future__ import annotations

from harness_agent.providers.base import resolve_api_key
from harness_agent.providers.openai_compat import OpenAIProvider

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider(OpenAIProvider):
    name = "deepseek"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            api_key=api_key or resolve_api_key("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
        )
