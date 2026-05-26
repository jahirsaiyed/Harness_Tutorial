"""Core42 Compass provider — OpenAI-compatible API at api.core42.ai."""

from __future__ import annotations

from harness_agent.providers.base import resolve_api_key
from harness_agent.providers.openai_compat import OpenAIProvider

COMPASS_BASE_URL = "https://api.core42.ai/v1"


class CompassProvider(OpenAIProvider):
    name = "compass"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            api_key=api_key or resolve_api_key("COMPASS_API_KEY"),
            base_url=COMPASS_BASE_URL,
        )
