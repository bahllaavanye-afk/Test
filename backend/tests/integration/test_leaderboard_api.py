"""Integration tests for the leaderboard API endpoints.

Covers the authenticated happy path for the handlers added alongside the CI
fix (the router previously had no route handlers at all). Exercises auth,
the per-strategy entry build, rank assignment (mutating the Pydantic model),
and serialization end-to-end — not just the 404/auth guard.
"""
from __future__ import annotations

import uuid

import pytest

_PASSWORD = "Sup3r-Secret-Pw!"


async def _auth_headers(client) -> dict[str, str]:
    """Register a fresh user and return an Authorization header."""
    email = f"lb-{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": _PASSWORD}
    )
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _seed_strategies(names: list[str]) -> None:
    """Insert Strategy rows directly so the leaderboard has something to rank."""
    from app.database import AsyncSessionLocal
    from app.models.strategy import Strategy

    async with AsyncSessionLocal() as s:
        for i, name in enumerate(names):
            s.add(
                Strategy(
                    name=name,
                    display_name=name.replace("_", " ").title(),
                    market_type="equity",
                    strategy_type="manual",
                    risk_bucket="directional",
                    symbols=["SPY"],
                    is_enabled=(i == 0),  # first one enabled, rest disabled
                )
            )
        await s.commit()


async def test_entries_requires_auth(client):
    r = await client.get("/api/v1/leaderboard/entries")
    assert r.status_code in (401, 403)


async def test_entries_happy_path_ranks_strategies(client):
    headers = await _auth_headers(client)
    names = [f"lb_strat_{uuid.uuid4().hex[:8]}" for _ in range(2)]
    await _seed_strategies(names)

    r = await client.get("/api/v1/leaderboard/entries", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body, list)

    by_name = {e["name"]: e for e in body}
    for name in names:
        assert name in by_name, f"{name} missing from leaderboard"
        entry = by_name[name]
        # identity fields populated, rank assigned, metric blocks present (may be null)
        for key in ("id", "market_type", "strategy_type", "risk_bucket",
                    "is_enabled", "symbols", "rank"):
            assert key in entry
        assert entry["rank"] >= 1

    # ranks are a contiguous 1..N sequence (assigned after sorting)
    ranks = sorted(e["rank"] for e in body)
    assert ranks == list(range(1, len(body) + 1))


async def test_summary_happy_path(client):
    headers = await _auth_headers(client)
    names = [f"lb_sum_{uuid.uuid4().hex[:8]}" for _ in range(2)]
    await _seed_strategies(names)

    r = await client.get("/api/v1/leaderboard/summary", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("total_strategies", "running_count", "avg_sharpe",
                "best_strategy", "total_paper_pnl", "total_live_pnl"):
        assert key in body
    assert body["total_strategies"] >= len(names)
    assert body["running_count"] >= 1  # we enabled at least one
