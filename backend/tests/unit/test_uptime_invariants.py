"""Enforce the no-downtime invariants so they can't silently regress.

Answers "no downtime?" — kept alive by TWO independent layers: an external
GitHub-Actions cron pinging /health (survives Render sleep), and an in-app
self-ping. These static checks fail loudly if either is removed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# Resolve the repository root; fallback to current directory if resolution fails.
try:
    ROOT: Optional[Path] = Path(__file__).resolve().parents[3]
except Exception:
    ROOT = Path('.')


def _safe_read(path: Optional[Path]) -> str:
    """Read a file safely.

    Returns an empty string if the path is None, does not exist, or any I/O error occurs.
    This guards against None inputs, missing files, and other edge cases without
    raising exceptions that would abort the test suite.
    """
    if not path:
        return ""
    try:
        return path.read_text()
    except Exception:
        return ""


def test_external_keepalive_pings_health_every_5_min():
    ka_content = _safe_read(ROOT / ".github" / "workflows" / "keep-alive.yml")
    # Guard against empty content which could happen if the file is missing.
    assert ka_content, "keep-alive workflow file missing or unreadable"
    assert (
        'cron: "*/5 * * * *"' in ka_content or "*/5 * * * *" in ka_content
    ), "keep-alive must run every 5 minutes"
    assert "/health" in ka_content, "keep-alive must hit the /health endpoint"


def test_in_app_self_ping_job_registered():
    sched_content = _safe_read(ROOT / "backend" / "app" / "tasks" / "scheduler.py")
    assert sched_content, "scheduler file missing or unreadable"
    assert (
        '"self_ping"' in sched_content or "self_ping" in sched_content
    ), "scheduler must register the self_ping job"
    assert (
        "RENDER_EXTERNAL_URL" in sched_content
    ), "self-ping must target the public Render URL"


def test_health_endpoint_exists():
    main_content = _safe_read(ROOT / "backend" / "app" / "main.py")
    assert main_content, "main.py missing or unreadable"
    assert (
        '"/health"' in main_content
    ), "backend must expose a /health endpoint"


def test_bot_runner_and_exit_checker_are_scheduled():
    """No-downtime for trading: the runner + exit checker must be wired in."""
    sched_content = _safe_read(ROOT / "backend" / "app" / "tasks" / "scheduler.py")
    assert sched_content, "scheduler file missing or unreadable"
    assert (
        "bot_runner_ignition" in sched_content
    ), "bot runner ignition job must exist"
    assert (
        "bot_exit_checker" in sched_content
    ), "bot exit checker job must exist (positions must close)"