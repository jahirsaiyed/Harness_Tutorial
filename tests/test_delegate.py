"""Delegate tool registration."""

import os
from pathlib import Path

os.environ.setdefault("HARNESS_AGENT_HOME", str(Path(__file__).resolve().parents[1] / "labs"))


def test_delegate_registered():
    from harness_agent.agent import AIAgent

    agent = AIAgent()
    assert "delegate_subagent" in agent.registry.list_available()
