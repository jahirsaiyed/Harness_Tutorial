"""Plugin discovery — register tools from plugins/ directory."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from harness_agent.config import get_config
from harness_agent.tools.registry import get_registry


def discover_plugins() -> list[str]:
    cfg = get_config()
    plugins_dir = cfg.home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    loaded: list[str] = []
    for path in plugins_dir.glob("*.py"):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if not spec or not spec.loader:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "register"):
            mod.register(get_registry())
            loaded.append(path.stem)
    return loaded
