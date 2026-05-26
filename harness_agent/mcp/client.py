"""MCP client — register remote tools into Harness Agent registry."""

from __future__ import annotations

import asyncio
from typing import Any

from harness_agent.observations import wrap_result
from harness_agent.tools.registry import get_registry


class MCPIntegration:
    """Attach tools from an MCP server (stdio) to the local registry."""

    def __init__(self) -> None:
        self._attached: list[str] = []

    async def connect_stdio(self, command: str, args: list[str] | None = None) -> list[str]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError("Install mcp package to use MCP integration") from exc

        params = StdioServerParameters(command=command, args=args or [])
        registry = get_registry()
        attached: list[str] = []

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                for tool in tools.tools:
                    name = f"mcp_{tool.name}"

                    def make_handler(tool_name: str, sess: ClientSession = session) -> Any:
                        async def _call(**kwargs: Any) -> str:
                            result = await sess.call_tool(tool_name, kwargs)
                            text = ""
                            for block in result.content:
                                if hasattr(block, "text"):
                                    text += block.text
                            return wrap_result(status="success", summary=text[:500], detail=text)

                        return _call

                    async_handler = make_handler(tool.name)

                    def sync_handler(**kwargs: Any, _ah=async_handler) -> str:
                        return asyncio.run(_ah(**kwargs))

                    registry.register(
                        name=name,
                        description=tool.description or f"MCP tool {tool.name}",
                        parameters=tool.inputSchema or {"type": "object", "properties": {}},
                        handler=sync_handler,
                        toolset="mcp",
                    )
                    attached.append(name)
        self._attached.extend(attached)
        return attached

    def connect_stdio_sync(self, command: str, args: list[str] | None = None) -> list[str]:
        return asyncio.run(self.connect_stdio(command, args))
