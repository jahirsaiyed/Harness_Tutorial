"""Example Harness Agent plugin — registers a demo tool."""

from __future__ import annotations

from harness_agent.observations import wrap_result


def hello_plugin(name: str = "world") -> str:
    return wrap_result(status="success", summary=f"Hello, {name}!")


def register(registry) -> None:
    registry.register(
        name="hello_plugin",
        description="Demo plugin tool.",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
        handler=hello_plugin,
        toolset="plugins",
    )
