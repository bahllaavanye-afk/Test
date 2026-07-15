"""Per-bot activity endpoint (Options Alpha detail-view parity).

GET /bots/{id}/activity must return this bot's open paper positions (orders
tagged raw_payload.bot_id by the engine) and recent closed-trade history
(Trade rows attributed by strategy_name == bot.name) — honest empty lists
when the bot hasn't traded, never other bots' rows.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

_PASSWORD = "B0tAct!2026x"


async def _auth_headers(client) -> dict[str, str]:
    from tests.integration._auth_helper import auth_headers

    return await auth_headers(client, prefix="botact", password=_PASSWORD)


def _bot_payload(name: str) -> dict:
    return {
        "name": name,
        "description": "activity endpoint test",
        "symbol": "SPY",
        "market_type": "equity",
        "trigger": {"type": "schedule", "interval": "1h"},
        "conditions": [],
        "action": {"type": "open_long", "size_pct": 5.0},
        "exit_rules": [],
    }


async def _seed_activity(bot_id: str, bot_name: str) -> None:
    from app.database import AsyncSessionLocal
    from app.models.account import Account
    from app.models.order import Order
    from app.models.trade import Trade
    from app.models.user import User

    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        user = User(id=str(uuid.uuid4()), email=f"seed_{uuid.uuid4().hex[:8]}@x.io", hashed_password="x")
        db.add(user)
        acct = Account(id=str(uuid.uuid4()), user_id=user.id, broker="alpaca", label="A", mode="paper")
        db.add(acct)
        # one open position for THIS bot, one for another bot (must not leak)
        db.add(Order(
            id=str(uuid.uuid4()), account_id=acct.id, symbol="SPY", side="buy",
            order_type="market", status="paper",
            raw_payload={"bot_id": bot_id, "bot_name": bot_name, "entry_price": 500.0, "notional": 1000.0},
        ))
        db.add(Order(
            id=str(uuid.uuid4()), account_id=acct.id, symbol="QQQ", side="buy",
            order_type="market", status="paper",
            raw_payload={"bot_id": "someone-elses-bot", "entry_price": 400.0},
        ))
        # two closed trades for this bot
        for i, pnl in enumerate((25.0, -10.0)):
            db.add(Trade(
                id=str(uuid.uuid4()), account_id=acct.id, strategy_name=bot_name,
                symbol="SPY", side="buy", entry_price=100, exit_price=100 + pnl / 10,
                quantity=10, realized_pnl=pnl, fees=0,
                opened_at=now - timedelta(days=i + 1, hours=2),
                closed_at=now - timedelta(days=i + 1),
                hold_seconds=7200, raw_payload={"exit_reason": "take_profit"},
            ))
        await db.commit()


@pytest.mark.asyncio
async def test_bot_activity_empty_for_fresh_bot(client):
    headers = await _auth_headers(client)
    r = await client.post("/api/v1/bots/", json=_bot_payload(f"Act Fresh {uuid.uuid4().hex[:6]}"),
                          headers=headers)
    assert r.status_code == 201, r.text
    bot = r.json()

    r = await client.get(f"/api/v1/bots/{bot['id']}/activity", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["open_positions"] == []
    assert body["trade_history"] == []


@pytest.mark.asyncio
async def test_bot_activity_returns_own_positions_and_trades_only(client):
    headers = await _auth_headers(client)
    name = f"Act Bot {uuid.uuid4().hex[:6]}"
    r = await client.post("/api/v1/bots/", json=_bot_payload(name), headers=headers)
    assert r.status_code == 201, r.text
    bot = r.json()

    await _seed_activity(bot["id"], name)

    r = await client.get(f"/api/v1/bots/{bot['id']}/activity", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()

    assert len(body["open_positions"]) == 1          # the other bot's order is excluded
    pos = body["open_positions"][0]
    assert pos["symbol"] == "SPY" and pos["entry_price"] == 500.0

    assert len(body["trade_history"]) == 2
    assert body["trade_history"][0]["closed_at"] >= body["trade_history"][1]["closed_at"]
    assert {t["realized_pnl"] for t in body["trade_history"]} == {25.0, -10.0}
    assert body["trade_history"][0]["exit_reason"] == "take_profit"


@pytest.mark.asyncio
async def test_bot_activity_404_for_other_users_bot(client):
    headers = await _auth_headers(client)
    r = await client.get(f"/api/v1/bots/{uuid.uuid4()}/activity", headers=headers)
    assert r.status_code == 404
