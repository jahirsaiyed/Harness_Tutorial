"""Cron scheduler for agent jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness_agent.agent import AIAgent
from harness_agent.config import get_config


@dataclass
class CronJob:
    id: str
    prompt: str
    schedule_minutes: int
    last_run: str | None = None
    delivery_webhook: str | None = None


class CronScheduler:
    def __init__(self, jobs_path: Path | None = None) -> None:
        self.jobs_path = jobs_path or get_config().cron_jobs_path

    def load_jobs(self) -> list[CronJob]:
        if not self.jobs_path.is_file():
            return []
        data = json.loads(self.jobs_path.read_text(encoding="utf-8"))
        return [CronJob(**item) for item in data.get("jobs", [])]

    def save_jobs(self, jobs: list[CronJob]) -> None:
        self.jobs_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"jobs": [job.__dict__ for job in jobs]}
        self.jobs_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def tick(self) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        results: list[dict[str, Any]] = []
        jobs = self.load_jobs()
        agent = AIAgent(isolated=True)
        for job in jobs:
            last = datetime.fromisoformat(job.last_run) if job.last_run else None
            due = last is None or (now - last).total_seconds() >= job.schedule_minutes * 60
            if not due:
                continue
            try:
                turn = agent.run_conversation(job.prompt)
                text = turn.assistant_text
            except Exception as exc:  # noqa: BLE001
                text = f"[cron error] {exc}"
            job.last_run = now.isoformat()
            entry = {"job_id": job.id, "text": text}
            results.append(entry)
            if job.delivery_webhook:
                try:
                    import httpx

                    httpx.post(job.delivery_webhook, json=entry, timeout=10)
                except Exception:  # noqa: BLE001
                    pass
        self.save_jobs(jobs)
        return results
