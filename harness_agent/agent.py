"""AIAgent — core conversation loop for Harness Agent."""

from __future__ import annotations

import json
from typing import Any

from harness_agent.compression.summarize import compress_messages, estimate_tokens
from harness_agent.config import get_config
from harness_agent.learning.skill_writer import LearningLoop, maybe_write_skill
from harness_agent.prompt.builder import PromptBuilder
from harness_agent.providers.registry import get_provider_registry
from harness_agent.sessions.store import SessionStore
from harness_agent.tools import file_tools, registry, terminal_tool  # noqa: F401
from harness_agent.tools.registry import get_registry
from harness_agent.types import AgentTurnResult, Message, ToolCall

# Ensure optional tools register
try:
    import harness_agent.delegate  # noqa: F401
except ImportError:
    pass


class AIAgent:
    def __init__(self, *, isolated: bool = False, toolsets: list[str] | None = None) -> None:
        self.config = get_config()
        self.isolated = isolated
        self.toolsets = toolsets
        self.prompt_builder = PromptBuilder()
        self.registry = get_registry()
        self.sessions = SessionStore()
        self.learning = LearningLoop()
        self._ensure_tools_loaded()

    def _ensure_tools_loaded(self) -> None:
        import harness_agent.delegate  # noqa: F401
        import harness_agent.sessions.store  # noqa: F401 — registers search_sessions
        import harness_agent.skills.loader  # noqa: F401

    def run_conversation(
        self,
        user_input: str,
        *,
        session_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> AgentTurnResult:
        if not session_id:
            session_id = self.sessions.create_session(title=user_input[:60])

        history = [] if self.isolated else self.sessions.load_messages(session_id)
        system = self.prompt_builder.build_system_prompt()
        messages: list[Message] = [Message(role="system", content=system)]
        messages.extend(history)
        messages.append(Message(role="user", content=user_input))

        prov, resolved_model = get_provider_registry().resolve(provider, model)
        tool_call_count = 0
        had_error = False
        final_text = ""

        for _ in range(self.config.max_turns):
            messages = compress_messages(messages)
            schemas = self.registry.openai_schemas(self.toolsets)
            text, calls = prov.complete_with_tools(messages, schemas, model=resolved_model)

            if calls:
                assistant_tool_calls = []
                for tc in calls:
                    assistant_tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                    )
                asst = Message(role="assistant", content=text, tool_calls=assistant_tool_calls)
                messages.append(asst)
                if not self.isolated:
                    self.sessions.append_turn(session_id, asst)

                for tc in calls:
                    tool_call_count += 1
                    result = self.registry.dispatch(tc.name, tc.arguments)
                    if '"status": "error"' in result:
                        had_error = True
                    tool_msg = Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                    messages.append(tool_msg)
                    if not self.isolated:
                        self.sessions.append_turn(session_id, tool_msg)
                continue

            final_text = text or ""
            asst = Message(role="assistant", content=final_text)
            messages.append(asst)
            if not self.isolated:
                self.sessions.append_turn(session_id, asst)
            break

        if not self.isolated and self.learning.should_author_skill(tool_call_count, had_error):
            maybe_write_skill(
                skill_name=f"workflow-{session_id[:8]}",
                description=f"Auto skill from session with {tool_call_count} tool calls",
                body=f"User goal:\n{user_input}\n\nOutcome:\n{final_text}",
                tool_call_count=tool_call_count,
            )

        return AgentTurnResult(
            assistant_text=final_text,
            messages=messages,
            tool_call_count=tool_call_count,
            session_id=session_id,
        )
