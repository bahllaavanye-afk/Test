"""Full OA-style bot lifecycle: open with brackets → profit-take / stop-loss →
closed Trade with P&L. This is the machinery the user compares against real
Option Alpha positions, and `check_bot_exits` (the closing/profit-taking half)
had ZERO coverage before this file.

Everything runs through the real engine (`_create_paper_order`,
`check_bot_exits`) against the test DB; only the price feed is monkeypatched
so exits are deterministic.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

import app.bots.engine as engine_mod
from app.bots.engine import BotEngine, check_bot_exits
from app.models.bot import Bot
from app.models.order import Order
from app.models.trade import Trade
from app.schemas.bot import ActionConfig


async def _mk_bot(db, account_id: str, user_id: str, symbol: str = "SPY") -> Bot:
    bot = Bot(
        id=str(uuid.uuid4()), user_id=user_id, account_id=account_id,
        name=f"lifecycle-{uuid.uuid4().hex[:6]}", symbol=symbol,
        market_type="equity", trigger={"type": "schedule", "interval": "1h"},
        conditions=[], condition_logic="ALL",
        action={"type": "open_long", "size_pct": 5.0,
                "take_profit_pct": 5.0, "stop_loss_pct": 3.0},
        exit_rules=[], is_enabled=True,
    )
    db.add(bot)
    await db.commit()
    await db.refresh(bot)
    return bot


async def _setup(client):
    """Demo user + paper account via the seed-style path (fresh rows per test)."""
    from app.database import AsyncSessionLocal
    from app.models.account import Account
    from app.models.user import User

    uid = str(uuid.uuid4())
    async with AsyncSessionLocal() as db:
        db.add(User(id=uid, email=f"lc-{uid[:8]}@example.com", hashed_password="x"))
        await db.commit()
        acct = Account(user_id=uid, broker="alpaca", mode="paper", label="Paper")
        db.add(acct)
        await db.commit()
        await db.refresh(acct)
        return uid, acct.id


async def _open(db, bot: Bot, entry: float, side: str = "buy") -> Order:
    """Open through the REAL engine path so bracket math is the code under test."""
    action = ActionConfig(**bot.action)
    oid = await BotEngine()._create_paper_order(bot, action, entry, side, db)
    await db.commit()
    return (await db.execute(select(Order).where(Order.id == oid))).scalar_one()


def _fake_price(price: float):
    async def _f(symbol: str, market_type: str = "equity"):
        return price
    return _f


async def test_open_sets_oa_style_brackets(client):
    from app.database import AsyncSessionLocal
    uid, acct_id = await _setup(client)
    async with AsyncSessionLocal() as db:
        bot = await _mk_bot(db, acct_id, uid)
        order = await _open(db, bot, entry=100.0)
        assert order.status == "paper"
        assert float(order.take_profit_price) == 105.0   # +5% TP
        assert float(order.stop_loss_price) == 97.0      # −3% SL
        assert (order.raw_payload or {})["bot_name"] == bot.name


async def test_profit_target_closes_with_positive_pnl(client, monkeypatch):
    from app.database import AsyncSessionLocal
    uid, acct_id = await _setup(client)
    async with AsyncSessionLocal() as db:
        bot = await _mk_bot(db, acct_id, uid)
        await _open(db, bot, entry=100.0)

        monkeypatch.setattr(engine_mod, "_fetch_current_price", _fake_price(106.0))
        exits = await check_bot_exits(db)
        await db.commit()
        assert exits >= 1

        trade = (await db.execute(
            select(Trade).where(Trade.strategy_name == bot.name)
        )).scalar_one()
        assert float(trade.entry_price) == 100.0
        assert float(trade.exit_price) == 105.0          # filled AT the target
        assert float(trade.realized_pnl) > 0
        # THIS bot's position is no longer open
        left = (await db.execute(select(Order).where(Order.status == 'paper'))).scalars().all()
        assert not any((o.raw_payload or {}).get('bot_name') == bot.name for o in left)


async def test_stop_loss_closes_with_negative_pnl(client, monkeypatch):
    from app.database import AsyncSessionLocal
    uid, acct_id = await _setup(client)
    async with AsyncSessionLocal() as db:
        bot = await _mk_bot(db, acct_id, uid)
        await _open(db, bot, entry=100.0)

        monkeypatch.setattr(engine_mod, "_fetch_current_price", _fake_price(95.0))
        assert await check_bot_exits(db) >= 1
        await db.commit()

        trade = (await db.execute(
            select(Trade).where(Trade.strategy_name == bot.name)
        )).scalar_one()
        assert float(trade.exit_price) == 97.0           # filled AT the stop
        assert float(trade.realized_pnl) < 0


async def test_short_side_profit_target(client, monkeypatch):
    from app.database import AsyncSessionLocal
    uid, acct_id = await _setup(client)
    async with AsyncSessionLocal() as db:
        bot = await _mk_bot(db, acct_id, uid)
        await _open(db, bot, entry=100.0, side="sell")   # TP below, SL above

        monkeypatch.setattr(engine_mod, "_fetch_current_price", _fake_price(94.0))
        assert await check_bot_exits(db) >= 1
        await db.commit()
        trade = (await db.execute(
            select(Trade).where(Trade.strategy_name == bot.name)
        )).scalar_one()
        assert float(trade.exit_price) == 95.0           # 100 × (1 − 5%)
        assert float(trade.realized_pnl) > 0             # short profits on the way down


async def test_neutral_price_keeps_position_open(client, monkeypatch):
    from app.database import AsyncSessionLocal
    uid, acct_id = await _setup(client)
    async with AsyncSessionLocal() as db:
        bot = await _mk_bot(db, acct_id, uid)
        await _open(db, bot, entry=100.0)
        monkeypatch.setattr(engine_mod, "_fetch_current_price", _fake_price(101.0))
        await check_bot_exits(db)                        # inside the bracket → MY position survives
        order = (await db.execute(select(Order).where(
            Order.status == "paper",
        ))).scalars().all()
        assert any((o.raw_payload or {}).get("bot_name") == bot.name for o in order)


async def test_stale_position_expires_after_seven_days(client, monkeypatch):
    from app.database import AsyncSessionLocal
    uid, acct_id = await _setup(client)
    async with AsyncSessionLocal() as db:
        bot = await _mk_bot(db, acct_id, uid)
        order = await _open(db, bot, entry=100.0)
        order.created_at = datetime.now(timezone.utc) - timedelta(days=8)
        await db.commit()

        monkeypatch.setattr(engine_mod, "_fetch_current_price", _fake_price(100.5))
        assert await check_bot_exits(db) == 1            # safety expiry
        await db.commit()
        trade = (await db.execute(
            select(Trade).where(Trade.strategy_name == bot.name)
        )).scalar_one()
        assert trade is not None
