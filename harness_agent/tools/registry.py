"""Central tool registry — register, schema collection, dispatch."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from harness_agent.observations import wrap_exception, wrap_result

ToolFn = Callable[..., str]
CheckFn = Callable[[], bool]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolFn
    toolset: str = "default"
    check_fn: CheckFn | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._toolsets: dict[str, set[str]] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: ToolFn,
        toolset: str = "default",
        check_fn: CheckFn | None = None,
    ) -> None:
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            toolset=toolset,
            check_fn=check_fn,
        )
        self._toolsets.setdefault(toolset, set()).add(name)

    def list_available(self, toolsets: list[str] | None = None) -> list[str]:
        names = []
        for name, spec in self._tools.items():
            if toolsets and spec.toolset not in toolsets:
                continue
            if spec.check_fn and not spec.check_fn():
                continue
            names.append(name)
        return sorted(names)

    def openai_schemas(self, toolsets: list[str] | None = None) -> list[dict[str, Any]]:
        schemas = []
        for name in self.list_available(toolsets):
            spec = self._tools[name]
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
            )
        return schemas

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            return wrap_result(
                status="error",
                summary=f"Unknown tool: {name}",
                next_actions=["Use a registered tool name."],
            )
        spec = self._tools[name]
        if spec.check_fn and not spec.check_fn():
            return wrap_result(
                status="error",
                summary=f"Tool unavailable: {name}",
                next_actions=["Check configuration."],
            )
        try:
            return spec.handler(**arguments)
        except Exception as exc:  # noqa: BLE001
            return wrap_exception(exc)


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register(**kwargs: Any) -> None:
    get_registry().register(**kwargs)
