"""Unit tests for desk → Trades reconstruction (pure, no DB / no network).

Guards the P&L-loop keystone: desk paper fills placed on Alpaca with
``qe-{strategy}-{symbol}-{ts}`` client_order_ids must reconstruct into correct
closed round trips so every strategy (not just bots) feeds the leaderboard.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, root_validator, validator

from app.tasks.desk_trade_sync import (
    parse_strategy_from_coid,
    reconstruct_closed_trades,
)


class Trade(BaseModel):
    """Schema representing a fully‑closed trade reconstructed from Alpaca fills.

    Attributes
    ----------
    strategy_name: str
        The original strategy identifier (e.g. ``momentum``). Must be non‑empty.
    symbol: str
        Trading symbol of the instrument, e.g. ``SPY`` or ``BTC/USD``.
    side: str
        ``buy`` for a long entry, ``sell`` for a short entry. Must be either
        ``\"buy\"`` or ``\"sell\"``.
    entry_price: float
        Execution price of the opening order.
    exit_price: float
        Execution price of the closing order.
    quantity: float
        Absolute quantity transacted. Must be positive.
    realized_pnl: float
        Profit‑and‑loss realised on the trade.
    hold_seconds: int
        Holding period in seconds (difference between exit and entry timestamps).
    open_order_id: str
        Alpaca order identifier of the opening fill.
    close_order_id: str
        Alpaca order identifier of the closing fill.
    """

    strategy_name: str = Field(
        ...,
        description="The strategy that generated the trade.",
        examples=["momentum", "mean_reversion"],
    )
    symbol: str = Field(
        ...,
        description="Ticker or instrument identifier.",
        examples=["SPY", "BTC/USD"],
    )
    side: str = Field(
        ...,
        description="Side of the opening order: 'buy' for long, 'sell' for short.",
        examples=["buy", "sell"],
    )
    entry_price: float = Field(
        ...,
        description="Price at which the position was opened.",
        examples=[100.0],
    )
    exit_price: float = Field(
        ...,
        description="Price at which the position was closed.",
        examples=[110.0],
    )
    quantity: float = Field(
        ...,
        gt=0,
        description="Absolute number of shares/contracts filled.",
        examples=[10],
    )
    realized_pnl: float = Field(
        ...,
        description="Realised profit or loss of the trade.",
        examples=[100.0],
    )
    hold_seconds: int = Field(
        ...,
        ge=0,
        description="Duration the position was held, in seconds.",
        examples=[3600],
    )
    open_order_id: str = Field(
        ...,
        description="Alpaca order ID of the opening trade.",
        examples=["o1"],
    )
    close_order_id: str = Field(
        ...,
        description="Alpaca order ID of the closing trade.",
        examples=["o2"],
    )

    @validator("side")
    def _validate_side(cls, v: str) -> str:
        if v not in {"buy", "sell"}:
            raise ValueError("side must be either 'buy' or 'sell'")
        return v

    @validator("strategy_name")
    def _non_empty_strategy(cls, v: str) -> str:
        if not v:
            raise ValueError("strategy_name cannot be empty")
        return v

    @root_validator
    def _check_hold_and_prices(cls, values):
        entry = values.get("entry_price")
        exit_ = values.get("exit_price")
        hold = values.get("hold_seconds")
        if hold is not None and hold < 0:
            raise ValueError("hold_seconds cannot be negative")
        if entry is not None and exit_ is not None:
            # No strict ordering required; allow short trades where exit < entry.
            pass
        return values


def _order(
    coid: str,
    side: str,
    qty: float,
    price: float,
    ts: str,
    oid: str,
    status: str = "filled",
    symbol: str = "SPY",
) -> dict:
    """Helper to build a minimal Alpaca order dict used by the tests."""
    return {
        "id": oid,
        "client_order_id": coid,
        "symbol": symbol,
        "side": side,
        "filled_qty": str(qty),
        "filled_avg_price": str(price),
        "filled_at": ts,
        "status": status,
    }


# ── client_order_id parsing / un-truncation ──────────────────────────────────


def test_parse_strategy_basic():
    assert parse_strategy_from_coid("qe-momentum-SPY-1720000000") == "momentum"


def test_parse_strategy_untruncates_with_registry():
    # desk_order_placer truncates the strategy to 10 chars: mean_reversion → mean_rever
    coid = "qe-mean_rever-AAPL-1720000000"
    assert parse_strategy_from_coid(coid, {"mean_reversion", "momentum"}) == "mean_reversion"


def test_parse_strategy_ignores_non_desk_orders():
    assert parse_strategy_from_coid("bot-abc-123") is None
    assert parse_strategy_from_coid(None) is None
    assert parse_strategy_from_coid("") is None


def test_parse_strategy_ambiguous_prefix_falls_back_to_token():
    # Two registry names share the first 10 chars → keep the raw (truncated) token.
    coid = "qe-vol_of_vo-SPY-1720000000"
    got = parse_strategy_from_coid(coid, {"vol_of_vol_timing", "vol_of_vol_carry"})
    assert got == "vol_of_vo"


# ── round‑trip reconstruction ────────────────────────────────────────────────


def _validate_trades(trades: List[dict]) -> None:
    """Validate each trade dict against the :class:`Trade` schema."""
    for trade_dict in trades:
        Trade(**trade_dict)


def test_long_round_trip_pnl():
    orders = [
        _order("qe-momentum-SPY-1", "buy", 10, 100.0, "2026-07-06T14:00:00Z", "o1"),
        _order("qe-momentum-SPY-2", "sell", 10, 110.0, "2026-07-06T15:00:00Z", "o2"),
    ]
    trades = reconstruct_closed_trades(orders)
    _validate_trades(trades)
    assert len(trades) == 1
    t = trades[0]
    assert t["strategy_name"] == "momentum"
    assert t["side"] == "buy"
    assert t["entry_price"] == 100.0
    assert t["exit_price"] == 110.0
    assert t["quantity"] == 10
    assert t["realized_pnl"] == 100.0  # (110-100)*10
    assert t["hold_seconds"] == 3600
    assert t["close_order_id"] == "o2"
    assert t["open_order_id"] == "o1"


def test_short_round_trip_pnl():
    orders = [
        _order("qe-breakout-QQQ-1", "sell", 5, 200.0, "2026-07-06T14:00:00Z", "s1", symbol="QQQ"),
        _order("qe-breakout-QQQ-2", "buy", 5, 190.0, "2026-07-06T14:30:00Z", "s2", symbol="QQQ"),
    ]
    trades = reconstruct_closed_trades(orders)
    _validate_trades(trades)
    assert len(trades) == 1
    t = trades[0]
    assert t["side"] == "sell"
    assert t["realized_pnl"] == 50.0  # (200-190)*5 for a short
    assert t["hold_seconds"] == 1800


def test_open_only_position_yields_no_trade():
    orders = [_order("qe-momentum-SPY-1", "buy", 10, 100.0, "2026-07-06T14:00:00Z", "o1")]
    assert reconstruct_closed_trades(orders) == []


def test_partial_close_then_full_close():
    orders = [
        _order("qe-rsi_macd-SPY-1", "buy", 10, 100.0, "2026-07-06T14:00:00Z", "o1"),
        _order("qe-rsi_macd-SPY-2", "sell", 4, 105.0, "2026-07-06T15:00:00Z", "o2"),
        _order("qe-rsi_macd-SPY-3", "sell", 6, 108.0, "2026-07-06T16:00:00Z", "o3"),
    ]
    trades = reconstruct_closed_trades(orders)
    _validate_trades(trades)
    assert len(trades) == 2
    assert trades[0]["quantity"] == 4
    assert trades[0]["realized_pnl"] == 20.0  # (105-100)*4
    assert trades[1]["quantity"] == 6
    assert trades[1]["realized_pnl"] == 48.0  # (108-100)*6


def test_position_flip_closes_then_opens():
    # buy 10, then sell 15: closes the 10-long, opens a 5-short (no 2nd trade yet).
    orders = [
        _order("qe-supertrend-SPY-1", "buy", 10, 100.0, "2026-07-06T14:00:00Z", "o1"),
        _order("qe-supertrend-SPY-2", "sell", 15, 110.0, "2026-07-06T15:00:00Z", "o2"),
    ]
    trades = reconstruct_closed_trades(orders)
    _validate_trades(trades)
    assert len(trades) == 1
    assert trades[0]["quantity"] == 10
    assert trades[0]["side"] == "buy"
    assert trades[0]["realized_pnl"] == 100.0


def test_separate_symbols_and_strategies_dont_cross():
    orders = [
        _order("qe-momentum-SPY-1", "buy", 10, 100.0, "2026-07-06T14:00:00Z", "o1", symbol="SPY"),
        _order("qe-momentum-QQQ-2", "sell", 10, 100.0, "2026-07-06T14:05:00Z", "o2", symbol="QQQ"),
    ]
    # A SPY buy and a QQQ sell must not net against each other → no closed trades.
    assert reconstruct_closed_trades(orders) == []


def test_unfilled_orders_ignored():
    orders = [
        _order("qe-momentum-SPY-1", "buy", 10, 100.0, "2026-07-06T14:00:00Z", "o1"),
        _order("qe-momentum-SPY-2", "sell", 10, 110.0, "2026-07-06T15:00:00Z", "o2", status="canceled"),
    ]
    assert reconstruct_closed_trades(orders) == []


def test_crypto_symbol_round_trip():
    orders = [
        _order("qe-crypto_ada-BTC-1", "buy", 0.5, 60000.0, "2026-07-06T14:00:00Z", "c1", symbol="BTC/USD"),
        _order("qe-crypto_ada-BTC-2", "sell", 0.5, 62000.0, "2026-07-06T18:00:00Z", "c2", symbol="BTC/USD"),
    ]
    trades = reconstruct_closed_trades(orders, {"crypto_adaptive_trend"})
    _validate_trades(trades)
    assert len(trades) == 1
    assert trades[0]["symbol"] == "BTC/USD"
    assert trades[0]["strategy_name"] == "crypto_adaptive_trend"
    assert trades[0]["realized_pnl"] == 1000.0  # (62000-60000)*0.5


def test_out_of_order_fills_are_sorted():
    # Feed the sell first; the reconstructor must sort by fill time before matching.
    orders = [
        _order("qe-momentum-SPY-2", "sell", 10, 110.0, "2026-07-06T15:00:00Z", "o2"),
        _order("qe-momentum-SPY-1", "buy", 10, 100.0, "2026-07-06T14:00:00Z", "o1"),
    ]
    trades = reconstruct_closed_trades(orders)
    _validate_trades(trades)
    assert len(trades) == 1
    assert trades[0]["entry_price"] == 100.0
    assert trades[0]["exit_price"] == 110.0