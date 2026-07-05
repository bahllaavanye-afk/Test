"""Enforce the no-downtime invariants so they can't silently regress.

Answers "no downtime?" — kept alive by TWO independent layers: an external
GitHub-Actions cron pinging /health (survives Render sleep), and an in-app
self-ping. These static checks fail loudly if either is removed.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_external_keepalive_pings_health_every_5_min():
    ka = (ROOT / ".github" / "workflows" / "keep-alive.yml").read_text()
    assert 'cron: "*/5 * * * *"' in ka or "*/5 * * * *" in ka, "keep-alive must run every 5 minutes"
    assert "/health" in ka, "keep-alive must hit the /health endpoint"


def test_in_app_self_ping_job_registered():
    sched = (ROOT / "backend" / "app" / "tasks" / "scheduler.py").read_text()
    assert '"self_ping"' in sched or "self_ping" in sched, "scheduler must register the self_ping job"
    assert "RENDER_EXTERNAL_URL" in sched, "self-ping must target the public Render URL"


def test_health_endpoint_exists():
    main = (ROOT / "backend" / "app" / "main.py").read_text()
    assert '"/health"' in main, "backend must expose a /health endpoint"


def test_bot_runner_and_exit_checker_are_scheduled():
    """No-downtime for trading: the runner + exit checker must be wired in."""
    sched = (ROOT / "backend" / "app" / "tasks" / "scheduler.py").read_text()
    assert "bot_runner_ignition" in sched, "bot runner ignition job must exist"
    assert "bot_exit_checker" in sched, "bot exit checker job must exist (positions must close)"
