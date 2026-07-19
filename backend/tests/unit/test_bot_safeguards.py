"""Per-bot position-limit safeguards (Option Alpha 'Safeguards').

The bot must refuse to open once it hits max_open_positions or
max_daily_positions, and the no_position/position_exists conditions must reflect
real open-position state. No network, no DB — fakes the session and OHLCV.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from app.bots.engine import BotEngine


def _bot(action: dict, conditions=None):
    return SimpleNamespace(
        id="bot1", account_id="acc1", symbol="SPY", market_type="equity",
        conditions=conditions or [], condition_logic="ALL", action=action,
        name="Guard Bot",
    )


def _df():
    return pd.DataFrame({"close": [100.0, 101.0, 102.0], "high": [102.0] * 3,
                         "low": [99.0] * 3, "volume": [1000.0] * 3})


async def _run(eng, bot, monkeypatch, *, open_now=0, today=0):
    async def fake_fetch(sym, mt):
        return _df()

    async def fake_count(self, b, db):
        return open_now, today

    async def fake_stats(self, b, db, r):
        return None

    monkeypatch.setattr("app.bots.engine._fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(BotEngine, "_count_bot_positions", fake_count)
    monkeypatch.setattr(BotEngine, "_update_bot_stats", fake_stats)
    return await eng._evaluate_inner(bot, db=None)


@pytest.mark.asyncio
async def test_open_position_limit_blocks_new_entry(monkeypatch):
    eng = BotEngine()
    bot = _bot({"type": "open_long", "size_pct": 5.0, "max_open_positions": 1})
    # 1 already open, limit is 1 → must not open.
    res = await _run(eng, bot, monkeypatch, open_now=1)
    assert res.fired is False
    assert "Position limit reached" in res.reason
    assert res.orders_created == []


@pytest.mark.asyncio
async def test_daily_position_limit_blocks_new_entry(monkeypatch):
    eng = BotEngine()
    bot = _bot({"type": "open_long", "size_pct": 5.0, "max_daily_positions": 2})
    res = await _run(eng, bot, monkeypatch, open_now=0, today=2)
    assert res.fired is False
    assert "Daily position limit reached" in res.reason


@pytest.mark.asyncio
async def test_under_limit_still_opens(monkeypatch):
    eng = BotEngine()
    bot = _bot({"type": "open_long", "size_pct": 5.0, "max_open_positions": 3})

    async def fake_order(self, b, a, price, side, db):
        return "order-1"

    monkeypatch.setattr(BotEngine, "_create_paper_order", fake_order)
    res = await _run(eng, bot, monkeypatch, open_now=1)   # 1 < 3
    assert res.fired is True
    assert res.orders_created == ["order-1"]


@pytest.mark.asyncio
async def test_no_position_condition_reflects_real_state(monkeypatch):
    eng = BotEngine()
    bot = _bot({"type": "open_long", "size_pct": 5.0},
               conditions=[{"type": "no_position"}])
    # A position is open → no_position is False → conditions fail → no entry.
    res = await _run(eng, bot, monkeypatch, open_now=1)
    assert res.fired is False
    assert res.signal == "hold"


@pytest.mark.asyncio
async def test_count_bot_positions_filters_by_bot_and_day():
    eng = BotEngine()
    now = datetime.now(timezone.utc)
    old = now.replace(year=now.year - 1)
    orders = [
        SimpleNamespace(status="paper", raw_payload={"bot_id": "bot1"}, created_at=now),
        SimpleNamespace(status="paper", raw_payload={"bot_id": "bot1"}, created_at=old),
        SimpleNamespace(status="paper", raw_payload={"bot_id": "other"}, created_at=now),
    ]

    class _Scalars:
        def all(self):
            return orders

    class _Result:
        def scalars(self):
            return _Scalars()

    class _DB:
        async def execute(self, *a, **k):
            return _Result()

    open_now, today = await eng._count_bot_positions(SimpleNamespace(id="bot1"), _DB())
    assert open_now == 2       # two orders tagged bot1 (other bot excluded)
    assert today == 1          # only one of them created today
