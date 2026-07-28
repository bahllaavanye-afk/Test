"""Fractional SELL orders died at the broker, every run.

Alpaca permits fractional shares on the LONG side but rejects them on the
short side. Measured 2026-07-28 on a live desk run — 3 of 14 attempted orders:

    422 {"code":42210000,"message":"fractional orders cannot be sold short"}
    place_order failed EIDO sell
    place_order failed ORCL sell
    place_order failed UNG sell

Those signals had already cleared data fetch, ensembling, the confidence gate,
Kelly sizing and the risk manager. Dying at the broker is the most expensive
possible place to discover an unplaceable order, and it recurs every run — the
same shape was visible on COST the day before.

A SELL IS NOT ALWAYS A SHORT. Under the daily loss cap only risk-REDUCING
orders pass, and those are closes (`is_risk_reducing`: sell against a long).
Blindly flooring would strand a sub-1-share long forever, because floor(0.4)
is 0 and the position could never be closed. The held quantity decides:

    held >= qty   closing a long   -> fractional is legal, leave untouched
    otherwise     opens a short    -> floor to whole shares, skip if < 1

Crypto is exempt: fractional is legal there in both directions, and the desks
trade tiny fractions of BTC and enormous fractions of SHIB.
"""
from __future__ import annotations

import asyncio

import pytest

import desk_order_placer as dop


def _call(symbol, side, qty, is_crypto=False, held=None):
    """Run the guard with a stubbed position map."""
    original = dop._position_map_cache
    dop._position_map_cache = {} if held is None else {symbol: held}
    try:
        return asyncio.run(
            dop._equity_short_safe_qty(symbol, side, qty, is_crypto)
        )
    finally:
        dop._position_map_cache = original


# ── the regression ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("symbol,qty", [
    ("EIDO", 12.34), ("ORCL", 3.57), ("UNG", 45.02),
])
def test_the_live_rejected_orders_are_now_whole_shares(symbol, qty):
    """The exact symbols Alpaca rejected on 2026-07-28."""
    out = _call(symbol, "sell", qty)
    assert out == float(int(qty)), out
    assert out == int(out), "must be a whole number of shares"


def test_a_sub_one_share_short_is_skipped_not_sent():
    """floor(0.4) is 0, and a 0-qty order is a guaranteed broker error."""
    assert _call("ORCL", "sell", 0.4) is None


def test_an_already_whole_short_is_untouched():
    assert _call("ORCL", "sell", 5.0) == 5.0


# ── closes must keep fractional ──────────────────────────────────────────────

def test_selling_into_a_larger_long_keeps_the_fraction():
    """A close, not a short — fractional is legal and must survive."""
    assert _call("ORCL", "sell", 3.57, held=10.0) == 3.57


def test_selling_exactly_the_held_amount_keeps_the_fraction():
    assert _call("ORCL", "sell", 3.57, held=3.57) == 3.57


def test_a_sub_one_share_long_can_still_be_closed():
    """THE REASON this is not a blind floor.

    Flooring here would return 0/None and the position could never be closed.
    """
    assert _call("ORCL", "sell", 0.4, held=0.4) == 0.4


def test_selling_MORE_than_held_is_floored():
    """Partly a close, partly a short — the short part sets the rule."""
    assert _call("ORCL", "sell", 9.6, held=2.0) == 9.0


# ── exemptions ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("symbol,qty", [
    ("BTC/USD", 0.00031), ("SHIB/USD", 251754653.84), ("UNI/USD", 678.09),
])
def test_crypto_is_exempt_in_both_directions(symbol, qty):
    """Fractional crypto is legal, and the desks depend on it."""
    assert _call(symbol, "sell", qty, is_crypto=True) == qty


@pytest.mark.parametrize("qty", [12.34, 0.4, 5.0])
def test_buys_are_never_touched(qty):
    """Fractional LONGS are legal — this guard is short-side only."""
    assert _call("ORCL", "buy", qty) == qty


def test_a_crypto_buy_is_untouched():
    assert _call("BTC/USD", "buy", 0.00031, is_crypto=True) == 0.00031


# ── the cache ────────────────────────────────────────────────────────────────

def test_the_position_map_is_fetched_at_most_once(monkeypatch):
    """It is consulted per sell order; an uncached fetch would be one broker
    round trip per signal."""
    calls = {"n": 0}

    async def _fake():
        calls["n"] += 1
        return {"ORCL": 100.0}

    monkeypatch.setattr(dop, "_alpaca_position_map", _fake)
    monkeypatch.setattr(dop, "_position_map_cache", None, raising=False)

    async def _drive():
        return [await dop._cached_position_map() for _ in range(5)]

    maps = asyncio.run(_drive())
    assert calls["n"] == 1, f"fetched {calls['n']} times"
    assert all(m == {"ORCL": 100.0} for m in maps)
    dop._position_map_cache = None
