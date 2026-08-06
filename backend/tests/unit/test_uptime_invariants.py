"""Enforce the no-downtime invariants so they can't silently regress.

Answers "no downtime?" — kept alive by TWO independent layers: an external
GitHub-Actions cron pinging /health (survives Render sleep), and an in-app
self-ping. These static checks fail loudly if either is removed.
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]


def test_external_keepalive_pings_health_every_5_min():
    ka_path = ROOT / ".github" / "workflows" / "keep-alive.yml"
    ka = ka_path.read_text()
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


def test_keepalive_workflow_file_exists_and_not_empty():
    """Edge case: ensure the keep-alive workflow file is present and not empty."""
    ka_path = ROOT / ".github" / "workflows" / "keep-alive.yml"
    assert ka_path.is_file(), "keep-alive workflow file must exist"
    content = ka_path.read_text()
    assert content.strip(), "keep-alive workflow file must not be empty"


def test_health_endpoint_defined_with_get_decorator():
    """Boundary condition: health endpoint should be a GET route."""
    main_path = ROOT / "backend" / "app" / "main.py"
    main = main_path.read_text()
    # Look for FastAPI GET decorator for /health
    pattern = r'@(?:app|router)\.get\(\s*["\']\/health["\']\s*\)'
    assert re.search(pattern, main), "health endpoint must be defined with a GET decorator"


def test_self_ping_schedule_has_valid_cron_expression():
    """Edge case: self_ping job should have a sensible cron expression (e.g., every minute or five)."""
    sched_path = ROOT / "backend" / "app" / "tasks" / "scheduler.py"
    sched = sched_path.read_text()
    # Find the line where self_ping is scheduled
    matches = re.findall(r'self_ping.*cron\s*=\s*["\']([^"\']+)["\']', sched)
    assert matches, "self_ping job must specify a cron expression"
    # Validate that the cron expression follows the standard 5-field pattern
    cron_pattern = r'^\s*\*\/?\d*\s+\*\s+\*\s+\*\s+\*$'
    for expr in matches:
        assert re.match(cron_pattern, expr), f"self_ping cron expression '{expr}' is not a valid 5-field pattern"