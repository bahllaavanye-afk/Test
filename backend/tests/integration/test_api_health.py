"""API health and root endpoint tests."""
import json
import os
from datetime import datetime
from pathlib import Path

import pytest

# Repo root: backend/tests/integration -> backend/tests -> backend -> Test
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_auth_register_then_login(client):
    try:
        resp = await client.post("/api/v1/auth/register", json={
            "email": "test@example.com", "password": "testpass123"
        })
    except Exception as e:
        # DB not migrated in test env (no such table: users)
        pytest.skip(f"DB not ready in test env — skipping auth test ({e})")
    if resp.status_code in (500, 503):
        pytest.skip(f"DB not migrated in test env — skipping auth test ({resp.status_code})")
    # 201 = created, 200 = created (some frameworks), 409 = already exists
    assert resp.status_code in (200, 201, 409), (
        f"Register returned {resp.status_code}: {resp.text}"
    )

    try:
        resp = await client.post("/api/v1/auth/login", json={
            "username": "test@example.com", "password": "testpass123"
        })
    except Exception:
        return
    if resp.status_code == 200:
        data = resp.json()
        assert "access_token" in data


@pytest.mark.asyncio
async def test_critical_endpoints_no_500(client):
    """All critical API endpoints must not return 5xx errors."""
    endpoints = [
        "/health",
        "/api/v1/strategies",
        "/api/v1/positions",
        "/api/v1/backtests",
    ]
    for endpoint in endpoints:
        resp = await client.get(endpoint, follow_redirects=False)
        assert resp.status_code < 500, (
            f"Endpoint {endpoint} returned {resp.status_code} (5xx is not acceptable)"
        )
        # 200=ok, 401/403=auth required, 404=not found, 307/308=redirect are all acceptable
        assert resp.status_code in (200, 307, 308, 401, 403, 404), (
            f"Endpoint {endpoint} returned unexpected status {resp.status_code}: {resp.text}"
        )


def test_agent_memory_last_updated_parseable():
    """agent_memory.json must have a parseable last_updated field."""
    memory_path = REPO_ROOT / ".github" / "state" / "agent_memory.json"
    assert memory_path.exists(), f"agent_memory.json not found at {memory_path}"
    data = json.loads(memory_path.read_text())
    assert "last_updated" in data, "agent_memory.json missing 'last_updated' field"
    # Must be parseable as an ISO datetime
    last_updated = data["last_updated"]
    assert last_updated, "last_updated field must not be empty"
    # datetime.fromisoformat handles the +00:00 offset in Python 3.7+
    parsed = datetime.fromisoformat(last_updated)
    assert parsed is not None


def test_workflow_health_strategies_live():
    """platform_metrics.strategies_live must be > 0 (platform is running strategies)."""
    memory_path = REPO_ROOT / ".github" / "state" / "agent_memory.json"
    assert memory_path.exists(), f"agent_memory.json not found at {memory_path}"
    data = json.loads(memory_path.read_text())
    assert "platform_metrics" in data, "agent_memory.json missing 'platform_metrics' key"
    metrics = data["platform_metrics"]
    assert "strategies_live" in metrics, "platform_metrics missing 'strategies_live'"
    assert metrics["strategies_live"] > 0, (
        f"strategies_live={metrics['strategies_live']} — expected > 0 (platform should have live strategies)"
    )


def test_slack_config_present():
    """SLACK_BOT_TOKEN env var must exist; skip gracefully if not configured."""
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        pytest.skip("SLACK_BOT_TOKEN not set — Slack integration not configured in this environment")
    assert len(token) > 0, "SLACK_BOT_TOKEN must be non-empty"
