"""Analytics honesty + tearsheet-crash regression (runs on SQLite test DB).

Two guarantees:
  1. /analytics/tearsheet no longer 500s on a non-Postgres backend — it used
     func.date_trunc() (Postgres-only), which raised on SQLite. Now grouped in
     Python, so it returns real metrics (or a clean 404 when empty).
  2. /analytics/live-stats reports HONEST numbers derived from real trades
     instead of the old hardcoded Sharpe 2.1 / win 68% / drawdown 14.7 constants.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

_PASSWORD = "An@lyt1cs!2026"


async def _auth_and_account(client):
    """Register a user, create an Alpaca paper account for them, return (headers, account_id)."""
    from sqlalchemy import select

    from app.database import AsyncSessionLocal
    from app.models.account import Account
    from app.models.user import User

    email = f"analytics_{uuid.uuid4().hex[:10]}@example.com"
    r = await client.post("/api/v1/auth/register", json={"email": email, "password": _PASSWORD})
    if r.status_code in (500, 503):
        pytest.skip(f"Auth backend unavailable ({r.status_code})")
    assert r.status_code == 201, r.text
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    async with AsyncSessionLocal() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalar_one()
        acct = Account(
            id=str(uuid.uuid4()), user_id=user.id, broker="alpaca",
            label="Paper", mode="paper", is_active=True,
        )
        db.add(acct)
        await db.commit()
        return headers, acct.id


async def _seed_trades(account_id: str, pnls: list[float]):
    from app.database import AsyncSessionLocal
    from app.models.trade import Trade

    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        for i, pnl in enumerate(pnls):
            opened = now - timedelta(days=len(pnls) - i, hours=2)
            closed = now - timedelta(days=len(pnls) - i)
            db.add(Trade(
                id=str(uuid.uuid4()), account_id=account_id, strategy_name="momentum",
                symbol="SPY", side="buy", entry_price=100.0,
                exit_price=100.0 + pnl / 10.0, quantity=10.0, realized_pnl=pnl,
                fees=0.0, opened_at=opened, closed_at=closed,
                hold_seconds=7200, raw_payload={"source": "test"},
            ))
        await db.commit()


@pytest.mark.asyncio
async def test_tearsheet_does_not_500_on_sqlite(client):
    """The date_trunc crash regression: with real trades, tearsheet returns 200 + metrics."""
    headers, account_id = await _auth_and_account(client)
    await _seed_trades(account_id, [120.0, -40.0, 80.0, -20.0, 200.0])

    r = await client.get("/api/v1/analytics/tearsheet?days=90", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_trades"] == 5
    assert "sharpe" in body and body["sharpe"] is not None
    assert isinstance(body["equity_curve"], list) and len(body["equity_curve"]) >= 1
    assert isinstance(body["monthly_returns"], list)
    # win_rate = 3 wins / 5 = 0.6
    assert body["win_rate"] == pytest.approx(0.6, abs=0.001)


@pytest.mark.asyncio
async def test_tearsheet_clean_404_when_no_trades(client):
    headers, _ = await _auth_and_account(client)
    r = await client.get("/api/v1/analytics/tearsheet?days=90", headers=headers)
    # Empty is an honest 404, never a 500.
    assert r.status_code == 404, r.text


@pytest.mark.asyncio
async def test_live_stats_are_honest_not_hardcoded(client):
    """live-stats must reflect real trades, not the old 2.1 / 68 / 14.7 constants."""
    from sqlalchemy import delete

    from app.database import AsyncSessionLocal
    from app.models.trade import Trade

    # live-stats aggregates trades GLOBALLY (no account filter), so clear first
    # for a deterministic win-rate independent of other tests in the session.
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Trade))
        await db.commit()

    _, account_id = await _auth_and_account(client)
    # 4 wins, 1 loss → win_rate 80.0 (deliberately NOT the old fake 68.0).
    await _seed_trades(account_id, [50.0, 50.0, 50.0, 50.0, -50.0])

    r = await client.get("/api/v1/analytics/live-stats")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_trades"] >= 5
    assert body["win_rate_pct"] == 80.0            # computed, not the hardcoded 68.0
    assert body["sharpe_ratio"] != 2.1             # real per-trade Sharpe, not the constant
    assert body["strategy_count"] >= 1
    assert body["trading_mode"] in ("paper", "test", "live")


@pytest.mark.asyncio
async def test_live_stats_null_when_no_data(client):
    """With zero trades, perf metrics are null (UI shows '—'), never fabricated."""
    from app.database import AsyncSessionLocal
    from app.models.trade import Trade
    from sqlalchemy import delete

    async with AsyncSessionLocal() as db:
        await db.execute(delete(Trade))
        await db.commit()

    r = await client.get("/api/v1/analytics/live-stats")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_trades"] == 0
    assert body["sharpe_ratio"] is None
    assert body["win_rate_pct"] is None
