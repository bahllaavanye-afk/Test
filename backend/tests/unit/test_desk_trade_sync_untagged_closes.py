"""Positions closed by anything but the desk placer never became Trades.

Reported twice as "still empty trades", and my first explanation — "nothing has
round-tripped yet" — was WRONG. `recover_negative_cash` flattened **25
positions** on 2026-07-27 at 18:43, and `/api/v1/trades/` still returned `[]`
long after.

THE CAUSE. `reconstruct_closed_trades` began with:

    strat = parse_strategy_from_coid(o.get("client_order_id"), registry_names)
    if strat is None:
        continue

`parse_strategy_from_coid` returns None for any order that is not `qe-`
prefixed. But the flatten goes through `DELETE /v2/positions`, so **Alpaca
generates those closing orders itself** and they carry no `qe-` tag. The
backend's PositionMonitor exits are the same shape.

So every untagged close was dropped, and the opening `qe-` buy left a lot that
could NEVER close. No Trade row, ever — which also starved the P&L feedback
loop and the leaderboard that `compute_live_strategy_performance` drives.

THE RULE. An untagged fill closes open lots for its SYMBOL, oldest first,
across whichever strategies hold them — that is what actually happened at the
broker. Attribution stays with the strategy that OPENED the lot, because the
close did not originate from a strategy and cannot introduce one. Excess beyond
open inventory is discarded rather than opening an unattributed lot: inventing
a strategy for it would corrupt exactly the attribution the leaderboard reads.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.tasks.desk_trade_sync import reconstruct_closed_trades

T0 = datetime(2026, 7, 27, 17, 49, tzinfo=timezone.utc)


def _order(coid, symbol, side, qty, price, minutes, oid=None):
    return {
        "id": oid or f"o{minutes}-{symbol}-{side}",
        "client_order_id": coid,
        "symbol": symbol,
        "side": side,
        "status": "filled",
        "filled_qty": str(qty),
        "filled_avg_price": str(price),
        "filled_at": (T0 + timedelta(minutes=minutes)).isoformat(),
    }


# ── the regression ───────────────────────────────────────────────────────────

def test_a_broker_flatten_now_produces_a_closed_trade():
    """THE BUG: the Jul 27 shape — desk buys, broker flattens, no trade."""
    orders = [
        _order("qe-avellaneda-AAVE-1785", "AAVEUSD", "buy", 10, 100.0, 0),
        _order("f1e2d3c4-broker-generated", "AAVEUSD", "sell", 10, 105.0, 54),
    ]
    trades = reconstruct_closed_trades(orders, ["avellaneda_stoikov_mm"])
    assert len(trades) == 1, trades
    t = trades[0]
    assert t["strategy_name"] == "avellaneda_stoikov_mm"
    assert t["realized_pnl"] == pytest.approx(50.0)
    assert t["side"] == "buy"
    assert t["hold_seconds"] == 54 * 60


def test_a_null_client_order_id_also_closes():
    """Alpaca may omit the field entirely on server-generated liquidations."""
    orders = [
        _order("qe-vol_of_vol-SHIB-1785", "SHIBUSD", "buy", 1000, 0.00001, 0),
        _order(None, "SHIBUSD", "sell", 1000, 0.000012, 30),
    ]
    trades = reconstruct_closed_trades(orders, ["vol_of_vol_timing"])
    assert len(trades) == 1
    assert trades[0]["strategy_name"] == "vol_of_vol_timing"


def test_the_25_position_flatten_shape():
    """Many symbols opened by desks, all flattened at once by the broker."""
    syms = [f"SYM{i}USD" for i in range(25)]
    orders = []
    for i, s in enumerate(syms):
        orders.append(_order(f"qe-avellaneda-{s[:4]}-{i}", s, "buy", 10, 100.0, i))
    for i, s in enumerate(syms):
        orders.append(_order(f"broker-{i}", s, "sell", 10, 99.0, 60 + i))
    trades = reconstruct_closed_trades(orders, ["avellaneda_stoikov_mm"])
    assert len(trades) == 25
    assert all(t["realized_pnl"] == pytest.approx(-10.0) for t in trades)


# ── attribution must not be invented ─────────────────────────────────────────

def test_an_untagged_close_keeps_the_OPENING_strategys_name():
    """The close carries no strategy; the lot's owner must be preserved."""
    orders = [
        _order("qe-supertrend-GLD-1", "GLD", "buy", 5, 200.0, 0),
        _order("manual-close", "GLD", "sell", 5, 210.0, 10),
    ]
    trades = reconstruct_closed_trades(orders, ["supertrend"])
    assert trades[0]["strategy_name"] == "supertrend"


def test_an_untagged_fill_with_no_inventory_creates_nothing():
    """No open lot to close -> discard, never an unattributed lot."""
    orders = [_order("broker-only", "SPY", "sell", 5, 600.0, 0)]
    assert reconstruct_closed_trades(orders, ["supertrend"]) == []


def test_excess_beyond_inventory_is_discarded_not_opened():
    """Closing more than is held must not leave a phantom short lot."""
    orders = [
        _order("qe-supertrend-SPY-1", "SPY", "buy", 5, 600.0, 0),
        _order("broker-close", "SPY", "sell", 12, 610.0, 10),
        _order("broker-close-2", "SPY", "buy", 7, 620.0, 20),
    ]
    trades = reconstruct_closed_trades(orders, ["supertrend"])
    assert len(trades) == 1
    assert trades[0]["quantity"] == pytest.approx(5)
    assert trades[0]["realized_pnl"] == pytest.approx(50.0)


def test_untagged_closes_oldest_lot_first_across_strategies():
    """FIFO by open time, regardless of which strategy holds the lot."""
    orders = [
        _order("qe-alpha-XYZ-1", "XYZ", "buy", 3, 10.0, 0),
        _order("qe-beta-XYZ-2", "XYZ", "buy", 3, 12.0, 5),
        _order("broker-flat", "XYZ", "sell", 3, 20.0, 60),
    ]
    trades = reconstruct_closed_trades(orders, ["alpha", "beta"])
    assert len(trades) == 1
    assert trades[0]["strategy_name"] == "alpha", "oldest lot must close first"
    assert trades[0]["entry_price"] == pytest.approx(10.0)


def test_an_untagged_close_only_touches_its_own_symbol():
    orders = [
        _order("qe-alpha-AAA-1", "AAA", "buy", 2, 50.0, 0),
        _order("qe-alpha-BBB-2", "BBB", "buy", 2, 60.0, 1),
        _order("broker-flat", "AAA", "sell", 2, 55.0, 10),
    ]
    trades = reconstruct_closed_trades(orders, ["alpha"])
    assert len(trades) == 1
    assert trades[0]["symbol"] == "AAA"


# ── existing behaviour must be untouched ─────────────────────────────────────

def test_a_fully_tagged_round_trip_is_unchanged():
    orders = [
        _order("qe-alpha-SPY-1", "SPY", "buy", 4, 100.0, 0),
        _order("qe-alpha-SPY-2", "SPY", "sell", 4, 110.0, 30),
    ]
    trades = reconstruct_closed_trades(orders, ["alpha"])
    assert len(trades) == 1
    assert trades[0]["realized_pnl"] == pytest.approx(40.0)
    assert trades[0]["strategy_name"] == "alpha"


def test_an_open_position_still_produces_no_trade():
    orders = [_order("qe-alpha-SPY-1", "SPY", "buy", 4, 100.0, 0)]
    assert reconstruct_closed_trades(orders, ["alpha"]) == []


def test_unfilled_orders_are_still_ignored():
    o = _order("broker-x", "SPY", "sell", 4, 100.0, 0)
    o["status"] = "canceled"
    assert reconstruct_closed_trades([o], ["alpha"]) == []


def test_a_short_closed_by_the_broker_prices_pnl_correctly():
    orders = [
        _order("qe-alpha-SPY-1", "SPY", "sell", 2, 100.0, 0),
        _order("broker-flat", "SPY", "buy", 2, 90.0, 10),
    ]
    trades = reconstruct_closed_trades(orders, ["alpha"])
    assert len(trades) == 1
    assert trades[0]["side"] == "sell"
    assert trades[0]["realized_pnl"] == pytest.approx(20.0)
