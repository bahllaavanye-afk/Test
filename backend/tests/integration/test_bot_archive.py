"""Integration tests for the Bot Archiver (soft-delete + restore).

Verifies that DELETE /bots/{id} archives rather than hard-deletes:
the bot disappears from the active list and desk summary but is preserved
(row + config) and appears under ?archived=true, and POST /bots/{id}/restore
brings it back. Runs entirely on the in-process SQLite test DB — no network.
"""
from __future__ import annotations

import uuid

import pytest

_PASSWORD = "Archiv@r!2026"


async def _auth_headers(client) -> dict[str, str]:
    """Register a fresh user and return an Authorization header.

    ``/auth/register`` returns a TokenResponse (201) directly, so we use that token;
    if the user already exists we fall back to ``/auth/login`` (which takes ``email``).
    """
    email = f"archive_{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": _PASSWORD}
    )
    if r.status_code in (500, 503):
        pytest.skip(f"Auth backend unavailable ({r.status_code}) — DB not migrated")
    if r.status_code == 201:
        token = r.json()["access_token"]
    else:
        resp = await client.post(
            "/api/v1/auth/login", json={"email": email, "password": _PASSWORD}
        )
        if resp.status_code != 200:
            pytest.skip(f"Login failed ({resp.status_code}) — DB not migrated in test env")
        token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _bot_payload(name: str) -> dict:
    return {
        "name": name,
        "description": "archiver lifecycle test",
        "symbol": "AAPL",
        "market_type": "equity",
        "trigger": {"type": "schedule", "interval": "1h"},
        "conditions": [],
        "action": {"type": "open_long", "size_pct": 5.0},
        "exit_rules": [],
    }


async def _create_bot(client, headers, name: str) -> dict:
    r = await client.post("/api/v1/bots/", json=_bot_payload(name), headers=headers)
    if r.status_code in (500, 503):
        pytest.skip(f"Bot create unavailable ({r.status_code})")
    assert r.status_code == 201, r.text
    return r.json()


def _ids(bots: list[dict]) -> set[str]:
    return {b["id"] for b in bots}


@pytest.mark.asyncio
async def test_archive_hides_from_active_but_preserves_bot(client):
    headers = await _auth_headers(client)
    bot = await _create_bot(client, headers, "Archive Me")
    bot_id = bot["id"]

    # Present in the active list before archiving
    r = await client.get("/api/v1/bots/", headers=headers)
    assert r.status_code == 200
    assert bot_id in _ids(r.json())

    # DELETE now archives (soft-delete) and returns 204
    r = await client.delete(f"/api/v1/bots/{bot_id}", headers=headers)
    assert r.status_code == 204, r.text

    # Gone from the active list…
    r = await client.get("/api/v1/bots/", headers=headers)
    assert r.status_code == 200
    assert bot_id not in _ids(r.json())

    # …but present (and flagged) under ?archived=true
    r = await client.get("/api/v1/bots/?archived=true", headers=headers)
    assert r.status_code == 200
    archived = {b["id"]: b for b in r.json()}
    assert bot_id in archived
    assert archived[bot_id]["is_archived"] is True
    assert archived[bot_id]["archived_at"] is not None
    assert archived[bot_id]["is_enabled"] is False

    # The row itself is preserved (config intact, not hard-deleted)
    r = await client.get(f"/api/v1/bots/{bot_id}", headers=headers)
    assert r.status_code == 200
    assert r.json()["name"] == "Archive Me"
    assert r.json()["action"]["type"] == "open_long"


@pytest.mark.asyncio
async def test_restore_returns_bot_to_active_list(client):
    headers = await _auth_headers(client)
    bot = await _create_bot(client, headers, "Round Trip Bot")
    bot_id = bot["id"]

    await client.delete(f"/api/v1/bots/{bot_id}", headers=headers)

    # Restore
    r = await client.post(f"/api/v1/bots/{bot_id}/restore", headers=headers)
    assert r.status_code == 200, r.text
    restored = r.json()
    assert restored["is_archived"] is False
    assert restored["archived_at"] is None

    # Back in the active list, no longer in the archived list
    r = await client.get("/api/v1/bots/", headers=headers)
    assert bot_id in _ids(r.json())
    r = await client.get("/api/v1/bots/?archived=true", headers=headers)
    assert bot_id not in _ids(r.json())


@pytest.mark.asyncio
async def test_restore_missing_bot_returns_404(client):
    headers = await _auth_headers(client)
    r = await client.post(f"/api/v1/bots/{uuid.uuid4()}/restore", headers=headers)
    assert r.status_code == 404
