"""Backtest → Bot generator (OA 'Automate your strategy').

POST /bots/from-backtest/{run_id} must turn a *completed* backtest owned by the
caller into a bot that carries the backtest's symbol/interval, maps the strategy
family to real engine conditions, bakes provenance into the description, and is
created **disabled** for review. Guards: 404 unknown run, 409 not-done run.
Runs on the in-process SQLite test DB — no network.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pytest

_PASSWORD = "Bt2B0t!2026x"


async def _register(client) -> tuple[dict[str, str], str]:
    """Register a fresh user; return (auth header, email) so the test can find the user id."""
    email = f"bt2bot_{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": _PASSWORD})
    if r.status_code in (429, 500, 503):
        pytest.skip(f"auth unavailable ({r.status_code})")
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}, email


async def _user_id_for(email: str) -> str:
    from sqlalchemy import select

    from app.database import AsyncSessionLocal
    from app.models.user import User

    async with AsyncSessionLocal() as db:
        return (await db.execute(select(User).where(User.email == email))).scalar_one().id


async def _seed_backtest(user_id: str, *, strategy="rsi2_pullback", symbol="SPY",
                         interval="1h", status="done", with_result=True) -> str:
    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun, BacktestResult

    run_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as db:
        db.add(BacktestRun(
            id=run_id, user_id=user_id, strategy_name=strategy, symbol=symbol,
            interval=interval, start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
            params={}, status=status, created_at=datetime.now(timezone.utc),
        ))
        if with_result:
            db.add(BacktestResult(
                id=str(uuid.uuid4()), run_id=run_id, total_return=0.42,
                sharpe_ratio=1.8, win_rate=0.61, profit_factor=2.1, total_trades=50,
            ))
        await db.commit()
    return run_id


@pytest.mark.asyncio
async def test_create_bot_from_backtest_maps_and_disables(client):
    headers, email = await _register(client)
    user_id = await _user_id_for(email)
    run_id = await _seed_backtest(user_id)

    r = await client.post(f"/api/v1/bots/from-backtest/{run_id}", json={"size_pct": 4.0}, headers=headers)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["confidence"] == "mapped"           # rsi2 maps to a real RSI condition
    assert body["source_backtest"] == run_id
    bot = body["bot"]
    assert bot["symbol"] == "SPY"
    assert bot["is_enabled"] is False               # paper-first: created disabled for review
    assert bot["trigger"]["interval"] == "1h"       # carried from the backtest
    assert bot["template_id"] == f"backtest:{run_id}"
    assert "rsi2_pullback" in bot["description"] and "Sharpe 1.80" in bot["description"]
    assert bot["action"]["size_pct"] == 4.0
    # the mapped RSI condition is present
    assert any(c.get("indicator") == "rsi" for c in bot["conditions"])


@pytest.mark.asyncio
async def test_unmapped_strategy_is_approx(client):
    headers, email = await _register(client)
    user_id = await _user_id_for(email)
    run_id = await _seed_backtest(user_id, strategy="poly_binary_arb")

    r = await client.post(f"/api/v1/bots/from-backtest/{run_id}", json={}, headers=headers)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["confidence"] == "approx"
    assert "review before enabling" in body["bot"]["description"]


@pytest.mark.asyncio
async def test_missing_and_unfinished_backtests_rejected(client):
    headers, email = await _register(client)
    user_id = await _user_id_for(email)

    r = await client.post(f"/api/v1/bots/from-backtest/{uuid.uuid4()}", json={}, headers=headers)
    assert r.status_code == 404, r.text

    running = await _seed_backtest(user_id, status="running", with_result=False)
    r = await client.post(f"/api/v1/bots/from-backtest/{running}", json={}, headers=headers)
    assert r.status_code == 409, r.text
