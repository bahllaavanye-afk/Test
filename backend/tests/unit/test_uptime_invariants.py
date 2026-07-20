"""Enforce the no-downtime invariants so they can't silently regress.

Answers "no downtime?" — kept alive by TWO independent layers: an external
GitHub-Actions cron pinging /health (survives Render sleep), and an in-app
self-ping. These static checks fail loudly if either is removed.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[3]

# Cached file reads to avoid repeated disk I/O
@lru_cache(maxsize=None)
def _read_file(path: Path) -> str:
    return path.read_text()


def test_external_keepalive_pings_health_every_5_min():
    ka_path = ROOT / ".github" / "workflows" / "keep-alive.yml"
    ka = _read_file(ka_path)
    assert 'cron: "*/5 * * * *"' in ka or "*/5 * * * *" in ka, "keep-alive must run every 5 minutes"
    assert "/health" in ka, "keep-alive must hit the /health endpoint"


def test_in_app_self_ping_job_registered():
    sched_path = ROOT / "backend" / "app" / "tasks" / "scheduler.py"
    sched = _read_file(sched_path)
    assert '"self_ping"' in sched or "self_ping" in sched, "scheduler must register the self_ping job"
    assert "RENDER_EXTERNAL_URL" in sched, "self-ping must target the public Render URL"


def test_health_endpoint_exists():
    main_path = ROOT / "backend" / "app" / "main.py"
    main = _read_file(main_path)
    assert '"/health"' in main, "backend must expose a /health endpoint"


def test_bot_runner_and_exit_checker_are_scheduled():
    """No-downtime for trading: the runner + exit checker must be wired in."""
    sched_path = ROOT / "backend" / "app" / "tasks" / "scheduler.py"
    sched = _read_file(sched_path)
    assert "bot_runner_ignition" in sched, "bot runner ignition job must exist"
    assert "bot_exit_checker" in sched, "bot exit checker job must exist (positions must close)"