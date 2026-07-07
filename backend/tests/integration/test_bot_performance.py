"""Per-bot performance endpoint (Options Alpha graph parity).

GET /bots/{id}/performance must return an honest cumulative realized-P&L series
built from real Trade rows attributed to the bot (strategy_name == bot.name) —
empty series when the bot hasn't closed a position, never fabricated.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

_PASSWORD = "B0tPerf!2026x"


async def _auth_headers(client) -> dict[str, str]:
    from tests.integration._auth_helper import auth_headers

    return await auth_headers(client, prefix="botperf", password=_PASSWORD)


def _bot_payload(name: str) -> dict:
    return {
        "name": name,
        "description": "perf endpoint test",
        "symbol": "SPY",
        "market_type": "equity",
        "trigger": {"type": "schedule", "interval": "1h"},
        "conditions": [],
        "action": {"type": "open_long", "size_pct": 5.0},
        "exit_rules": [],
    }


async def _seed_bot_trades(bot_name: str, pnls: list[float]) -> None:
    from app.database import AsyncSessionLocal
    from app.models.account import Account
    from app.models.trade import Trade
    from app.models.user import User

    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        # any account satisfies the FK; make one with its own throwaway user
        user = User(id=str(uuid.uuid4()), email=f"seed_{uuid.uuid4().hex[:8]}@x.io", hashed_password="x")
        db.add(user)
        acct = Account(id=str(uuid.uuid4()), user_id=user.id, broker="alpaca", label="P", mode="paper")
        db.add(acct)
        for i, pnl in enumerate(pnls):
            db.add(Trade(
                id=str(uuid.uuid4()), account_id=acct.id, strategy_name=bot_name,
                symbol="SPY", side="buy", entry_price=100, exit_price=100 + pnl / 10,
                quantity=10, realized_pnl=pnl, fees=0,
                opened_at=now - timedelta(days=len(pnls) - i, hours=3),
                closed_at=now - timedelta(days=len(pnls) - i),
                hold_seconds=10800, raw_payload={},
            ))
        await db.commit()


@pytest.mark.asyncio
async def test_bot_performance_series_and_stats(client):
    headers = await _auth_headers(client)
    name = f"Perf Bot {uuid.uuid4().hex[:6]}"
    r = await client.post("/api/v1/bots/", json=_bot_payload(name), headers=headers)
    assert r.status_code == 201, r.text
    bot_id = r.json()["id"]

    await _seed_bot_trades(name, [100.0, -50.0, 200.0])

    r = await client.get(f"/api/v1/bots/{bot_id}/performance?days=30", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trades"] == 3
    assert body["total_pnl"] == 250.0
    assert body["win_rate"] == pytest.approx(2 / 3, abs=0.001)
    assert [p["cum_pnl"] for p in body["series"]] == [100.0, 50.0, 250.0]
    assert body["max_drawdown"] == 50.0


@pytest.mark.asyncio
async def test_bot_performance_empty_is_honest(client):
    headers = await _auth_headers(client)
    r = await client.post("/api/v1/bots/", json=_bot_payload(f"Empty Bot {uuid.uuid4().hex[:6]}"), headers=headers)
    assert r.status_code == 201, r.text
    r = await client.get(f"/api/v1/bots/{r.json()['id']}/performance", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["series"] == [] and body["trades"] == 0 and body["win_rate"] is None
