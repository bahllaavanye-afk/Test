"""Discord wiring invariants — every link in the chain, testable without Discord.

The empty-Discord saga exposed that each piece can silently regress: the
interactions endpoint, signature rejection, the channel structure, the
event-driven ops-sync, and the startup jobs. These tests pin all of them.
"""
from __future__ import annotations

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _read_file(path: Path | None) -> str:
    """Safely read a file, providing clear failures for None or missing files."""
    if path is None:
        pytest.fail("Provided path is None")
    if not path.is_file():
        pytest.fail(f"Expected file not found: {path}")
    content = path.read_text()
    if not content:
        pytest.fail(f"File is empty: {path}")
    return content


def test_interactions_endpoint_mounted():
    router_path = ROOT / "backend" / "app" / "api" / "v1" / "router.py"
    router = _read_file(router_path)
    assert "discord_interactions" in router, "Discord interactions router must be mounted"


def test_unsigned_interaction_is_rejected():
    """Ed25519 gate: a request without valid signature headers must never pass."""
    from starlette.testclient import TestClient

    from app.main import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/api/v1/discord/interactions", json={"type": 1})
    assert r.status_code == 401, f"unsigned interaction must 401, got {r.status_code}"


def test_channel_structure_covers_notify_channels():
    """Every channel notify.py routes to must exist in the setup structure."""
    setup_path = ROOT / ".github" / "scripts" / "discord_setup_channels.py"
    setup = _read_file(setup_path)

    channels = (
        "desk-equities",
        "desk-crypto",
        "desk-options",
        "desk-polymarket",
        "desk-fx-rates",
        "desk-stat-arb",
        "infra-alerts",
        "risk-alerts",
        "pnl-daily",
        "ci-failures",
        "engineering",
        "alpha-research",
        "leadership-summary",
    )
    if not channels:
        pytest.fail("Channel list is empty")
    for ch in channels:
        assert f'"{ch}"' in setup, f"channel setup is missing #{ch}"


def test_ops_sync_rides_ci_not_cron():
    """The cron-starvation fix: channel setup + key relay must run inside CI."""
    ci_path = ROOT / ".github" / "workflows" / "test.yml"
    ci = _read_file(ci_path)
    assert "ops-sync:" in ci, "CI must contain the ops-sync job"
    assert "discord_setup_channels.py" in ci, "ops-sync must create channels"
    assert "applications/$APP_ID/commands" in ci, "ops-sync must register slash commands"
    assert "continue-on-error: true" in ci, "ops-sync must never block a merge"


def test_backend_startup_jobs_registered():
    sched_path = ROOT / "backend" / "app" / "tasks" / "scheduler.py"
    sched = _read_file(sched_path)
    required_jobs = ("discord_channel_setup", "discord_command_registration", "hourly_standup")
    if not required_jobs:
        pytest.fail("Startup jobs list is empty")
    for job in required_jobs:
        assert job in sched, f"startup {job} job must exist"


def test_slash_commands_consistent_everywhere():
    """The 4 commands must match across handler, CI sync, and startup sync."""
    handler_path = ROOT / "backend" / "app" / "api" / "v1" / "discord_interactions.py"
    ci_path = ROOT / ".github" / "workflows" / "test.yml"
    sched_path = ROOT / "backend" / "app" / "tasks" / "scheduler.py"

    handler = _read_file(handler_path)
    ci = _read_file(ci_path)
    sched = _read_file(sched_path)

    commands = ("status", "pnl", "health", "run-bot")
    if not commands:
        pytest.fail("Command list is empty")
    for cmd in commands:
        assert cmd in handler, f"handler missing /{cmd}"
        # Remove whitespace for a stricter check but also allow raw presence
        ci_clean = ci.replace(" ", "")
        assert f'"name":"{cmd}"' in ci_clean or cmd in ci, f"CI sync missing /{cmd}"
        assert cmd in sched, f"startup sync missing /{cmd}"