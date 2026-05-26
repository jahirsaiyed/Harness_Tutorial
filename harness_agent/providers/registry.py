"""Provider registry and runtime resolution."""

from __future__ import annotations

import os

from harness_agent.providers.anthropic_compat import AnthropicProvider
from harness_agent.providers.base import BaseProvider
from harness_agent.providers.compass_compat import CompassProvider
from harness_agent.providers.deepseek_compat import DeepSeekProvider
from harness_agent.providers.openai_compat import OpenAIProvider


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> BaseProvider:
        if name not in self._providers:
            raise KeyError(f"Unknown provider: {name}")
        return self._providers[name]

    def resolve(self, provider_name: str | None = None, model: str | None = None) -> tuple[BaseProvider, str]:
        cfg_provider = provider_name or os.environ.get("HARNESS_DEFAULT_PROVIDER", "openai")
        cfg_model = model or os.environ.get("HARNESS_DEFAULT_MODEL", "gpt-4o-mini")
        return self.get(cfg_provider), cfg_model


_registry: ProviderRegistry | None = None


def get_provider_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
        _registry.register(OpenAIProvider())
        _registry.register(AnthropicProvider())
        _registry.register(DeepSeekProvider())
        _registry.register(CompassProvider())
    return _registry
