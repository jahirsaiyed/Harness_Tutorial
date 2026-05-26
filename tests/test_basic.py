"""Offline tests for Harness Agent tutorial."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("HARNESS_AGENT_HOME", str(Path(__file__).resolve().parents[1] / "labs"))


def test_config_home():
    from harness_agent.config import get_config

    cfg = get_config()
    assert cfg.home.exists()


def test_tool_registry():
    from harness_agent.tools.registry import get_registry

    reg = get_registry()
    assert "read_file" in reg.list_available()


def test_session_store():
    from harness_agent.sessions.store import SessionStore

    store = SessionStore()
    sid = store.create_session("test")
    store.append_turn(sid, __import__("harness_agent.types", fromlist=["Message"]).Message(role="user", content="hi"))
    msgs = store.load_messages(sid)
    assert len(msgs) == 1


def test_prompt_builder():
    from harness_agent.prompt.builder import PromptBuilder

    p = PromptBuilder().build_system_prompt()
    assert "Harness Agent" in p


def test_compression():
    from harness_agent.compression.summarize import compress_messages
    from harness_agent.types import Message

    msgs = [Message(role="user", content="x" * 5000) for _ in range(20)]
    out = compress_messages(msgs, max_tokens=1000)
    assert len(out) < len(msgs)


def test_cron_load():
    from harness_agent.cron.scheduler import CronScheduler

    jobs = CronScheduler().load_jobs()
    assert len(jobs) >= 1
