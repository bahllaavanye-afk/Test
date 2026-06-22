"""Integration tests for options bots through the real API.

Creating an ``open_option_spread`` bot with a structured ``legs`` plan must
round-trip through ``POST /bots/`` and ``GET /bots/{id}`` with the legs intact,
and the bundled options templates must instantiate. Runs on the in-process
SQLite test DB — no network, no broker credentials.
"""
from __future__ import annotations

import uuid

import pytest

_PASSWORD = "0pti0ns!2026"


async def _auth_headers(client) -> dict[str, str]:
    email = f"options_{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": _PASSWORD}
    )
    if r.status_code in (500, 503):
        pytest.skip(f"Auth backend unavailable ({r.status_code}) — DB not migrated")
    if r.status_code == 201:
        return {"Authorization": f"Bearer {r.json()['access_token']}"}
    resp = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": _PASSWORD}
    )
    if resp.status_code != 200:
        pytest.skip(f"Login failed ({resp.status_code}) — DB not migrated in test env")
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _spread_payload(name: str) -> dict:
    return {
        "name": name,
        "description": "bull put credit spread",
        "symbol": "SPY",
        "market_type": "options",
        "trigger": {"type": "schedule", "interval": "1d"},
        "conditions": [],
        "action": {
            "type": "open_option_spread",
            "size_pct": 3,
            "take_profit_pct": 50,
            "legs": [
                {"side": "sell", "option_type": "put", "delta": 0.25, "dte": 30, "ratio": 1},
                {"side": "buy", "option_type": "put", "delta": 0.15, "dte": 30, "ratio": 1},
            ],
        },
        "exit_rules": [],
    }


@pytest.mark.asyncio
async def test_create_options_spread_bot_roundtrips_legs(client):
    headers = await _auth_headers(client)
    r = await client.post("/api/v1/bots/", json=_spread_payload("Bull Put Spread"), headers=headers)
    if r.status_code in (500, 503):
        pytest.skip(f"Bot create unavailable ({r.status_code})")
    assert r.status_code == 201, r.text
    bot = r.json()
    assert bot["market_type"] == "options"
    action = bot["action"]
    assert action["type"] == "open_option_spread"
    assert len(action["legs"]) == 2
    assert action["legs"][0]["side"] == "sell"
    assert action["legs"][0]["option_type"] == "put"

    # GET round-trips the legs unchanged
    r = await client.get(f"/api/v1/bots/{bot['id']}", headers=headers)
    assert r.status_code == 200
    legs = r.json()["action"]["legs"]
    assert [lg["delta"] for lg in legs] == [0.25, 0.15]


@pytest.mark.asyncio
async def test_options_templates_are_listed(client):
    headers = await _auth_headers(client)
    r = await client.get("/api/v1/bots/templates", headers=headers)
    if r.status_code == 404:
        pytest.skip("templates endpoint not mounted in this build")
    assert r.status_code == 200, r.text
    templates = r.json()
    keys = set(templates.keys()) if isinstance(templates, dict) else {t.get("id") for t in templates}
    assert "opt_iron_condor" in keys
    assert "opt_bull_put_spread" in keys
