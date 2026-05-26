"""Context compression — summarize middle turns."""

from __future__ import annotations

from harness_agent.types import Message


def estimate_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        total += len(m.content or "") // 4
        if m.tool_calls:
            total += 100 * len(m.tool_calls)
    return total


def compress_messages(messages: list[Message], *, max_tokens: int = 12000) -> list[Message]:
    if estimate_tokens(messages) <= max_tokens:
        return messages
    if len(messages) <= 4:
        return messages
    head = messages[:2]
    tail = messages[-4:]
    middle = messages[2:-4]
    summary_bits = []
    for m in middle:
        role = m.role
        snippet = (m.content or "")[:200]
        summary_bits.append(f"{role}: {snippet}")
    summary = Message(
        role="user",
        content="[Compressed middle history]\n" + "\n".join(summary_bits),
    )
    return head + [summary] + tail
