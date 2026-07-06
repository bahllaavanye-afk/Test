"""Discord wiring invariants — every link in the chain, testable without Discord.

The empty-Discord saga exposed that each piece can silently regress: the
interactions endpoint, signature rejection, the channel structure, the
event-driven ops-sync, and the startup jobs. These tests pin all of them.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_interactions_endpoint_mounted():
    router = (ROOT / "backend" / "app" / "api" / "v1" / "router.py").read_text()
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
    setup = (ROOT / ".github" / "scripts" / "discord_setup_channels.py").read_text()
    for ch in ("desk-equities", "desk-crypto", "desk-options", "desk-polymarket",
               "desk-fx-rates", "desk-stat-arb", "infra-alerts", "risk-alerts",
               "pnl-daily", "ci-failures", "engineering", "alpha-research",
               "leadership-summary"):
        assert f'"{ch}"' in setup, f"channel setup is missing #{ch}"


def test_ops_sync_rides_ci_not_cron():
    """The cron-starvation fix: channel setup + key relay must run inside CI."""
    ci = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "ops-sync:" in ci, "CI must contain the ops-sync job"
    assert "discord_setup_channels.py" in ci, "ops-sync must create channels"
    assert "applications/$APP_ID/commands" in ci, "ops-sync must register slash commands"
    assert "continue-on-error: true" in ci, "ops-sync must never block a merge"


def test_backend_startup_jobs_registered():
    sched = (ROOT / "backend" / "app" / "tasks" / "scheduler.py").read_text()
    assert "discord_channel_setup" in sched, "startup channel creation job must exist"
    assert "discord_command_registration" in sched, "startup command registration job must exist"
    assert "hourly_standup" in sched, "hourly standup job must exist"


def test_slash_commands_consistent_everywhere():
    """The 4 commands must match across handler, CI sync, and startup sync."""
    handler = (ROOT / "backend" / "app" / "api" / "v1" / "discord_interactions.py").read_text()
    ci = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    sched = (ROOT / "backend" / "app" / "tasks" / "scheduler.py").read_text()
    for cmd in ("status", "pnl", "health", "run-bot"):
        assert cmd in handler, f"handler missing /{cmd}"
        assert f'"name":"{cmd}"' in ci.replace(" ", "") or cmd in ci, f"CI sync missing /{cmd}"
        assert cmd in sched, f"startup sync missing /{cmd}"
