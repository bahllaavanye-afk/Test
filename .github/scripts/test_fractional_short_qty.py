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


# ── the market-replacement path ──────────────────────────────────────────────
# The first fix covered the LIMIT branch only, and one 422 survived the next
# live run:
#
#   · UNG sell 37.27 -> 37 whole shares      <- limit path, correctly fixed
#   ↻ limit unfilled after 20s — replaced with market
#   ⚠ 422 {"code":42210000,"message":"fractional orders cannot be sold short"}
#
# `_ensure_filled` cancel-replaces an unfilled limit by calling `_place_order`
# with NO limit price, which takes the equity market branch. That branch sent a
# NOTIONAL order, so Alpaca derived the share count itself — fractionally — and
# rejected it short. A short-side equity market order must carry an explicit
# whole `qty`.


def _place_body(monkeypatch, symbol, side, notional, *, price=10.0, held=None):
    """Capture the body `_place_order` would POST, with no limit price."""
    import asyncio

    posted: dict = {}

    async def _fake_post(path, body):
        posted.update(body)
        return {"id": "x", "status": "accepted"}

    async def _fake_price(_sym):
        return price

    monkeypatch.setattr(dop, "_alpaca_post", _fake_post)
    monkeypatch.setattr(dop, "_equity_last_price", _fake_price)
    monkeypatch.setattr(dop, "_position_map_cache",
                        {} if held is None else {symbol: held}, raising=False)
    asyncio.run(dop._place_order(symbol, side, notional))
    dop._position_map_cache = None
    return posted


def test_a_short_market_order_carries_whole_qty_not_notional(monkeypatch):
    """THE SURVIVING BUG: notional -> Alpaca derives a fraction -> 422."""
    body = _place_body(monkeypatch, "UNG", "sell", 372.7, price=10.0)
    assert body.get("qty") == "37.0", body
    assert "notional" not in body, "notional lets Alpaca pick a fractional qty"


def test_a_buy_market_order_still_uses_notional(monkeypatch):
    """Fractional longs are legal — do not lose notional precision on buys."""
    body = _place_body(monkeypatch, "UNG", "buy", 372.7, price=10.0)
    assert body.get("notional") == "372.7", body
    assert "qty" not in body


def test_a_short_market_order_under_one_share_is_not_sent(monkeypatch):
    """floor gives 0 — skip rather than post a guaranteed rejection."""
    body = _place_body(monkeypatch, "UNG", "sell", 4.0, price=10.0)
    assert body == {}, body


def test_selling_into_a_long_keeps_notional_precision(monkeypatch):
    """A close is fractional-legal, so it need not be whole-shared."""
    body = _place_body(monkeypatch, "UNG", "sell", 37.0, price=10.0, held=100.0)
    assert body.get("qty") == "3.7", body


def test_an_unavailable_price_falls_back_to_notional(monkeypatch):
    """Fail-soft: worse for shorts, but never worse than not ordering."""
    import asyncio

    posted: dict = {}

    async def _fake_post(path, body):
        posted.update(body)
        return {"id": "x"}

    async def _no_price(_sym):
        return None

    monkeypatch.setattr(dop, "_alpaca_post", _fake_post)
    monkeypatch.setattr(dop, "_equity_last_price", _no_price)
    monkeypatch.setattr(dop, "_position_map_cache", {}, raising=False)
    asyncio.run(dop._place_order("UNG", "sell", 372.7))
    dop._position_map_cache = None
    assert posted.get("notional") == "372.7", posted


# ── non-shortable assets ─────────────────────────────────────────────────────
# Whole-sharing the quantity does not help when the ASSET cannot be shorted at
# all. Seen on EIDO in two consecutive runs, AFTER the qty was correctly
# rounded to 44 whole shares:
#
#   · EIDO sell 44.23 -> 44 whole shares (fractional shorts are rejected)
#   ⚠ 422 {"code":42210000,"message":"asset \"EIDO\" cannot be sold short"}
#
# Check the asset, not just the number.


def _call_short(monkeypatch, symbol, qty, *, shortable=True, held=None):
    monkeypatch.setattr(dop, "_shortable_cache", {symbol: shortable}, raising=False)
    monkeypatch.setattr(dop, "_position_map_cache",
                        {} if held is None else {symbol: held}, raising=False)
    out = asyncio.run(dop._equity_short_safe_qty(symbol, "sell", qty, False))
    dop._position_map_cache = None
    dop._shortable_cache = {}
    return out


def test_a_non_shortable_asset_is_skipped(monkeypatch):
    """THE BUG: EIDO, rejected twice after the qty was already correct."""
    assert _call_short(monkeypatch, "EIDO", 44.23, shortable=False) is None


def test_a_shortable_asset_is_still_whole_shared(monkeypatch):
    assert _call_short(monkeypatch, "UNG", 37.27, shortable=True) == 37.0


def test_a_non_shortable_asset_can_still_be_SOLD_to_close(monkeypatch):
    """Closing a long is not a short sale — shortability is irrelevant.

    Blocking this would strand every long in a non-shortable asset.
    """
    assert _call_short(monkeypatch, "EIDO", 10.0, shortable=False, held=50.0) == 10.0


def test_an_unknown_asset_fails_soft_to_shortable(monkeypatch):
    """A lookup blip must not silently stop the desks selling."""
    async def _boom(_path, *a, **k):
        raise RuntimeError("assets endpoint down")

    monkeypatch.setattr(dop, "_alpaca_get", _boom)
    monkeypatch.setattr(dop, "_shortable_cache", {}, raising=False)
    assert asyncio.run(dop._is_shortable("ANY")) is True


def test_the_shortable_lookup_is_cached(monkeypatch):
    calls = {"n": 0}

    async def _fake(_path, *a, **k):
        calls["n"] += 1
        return {"shortable": True, "tradable": True}

    monkeypatch.setattr(dop, "_alpaca_get", _fake)
    monkeypatch.setattr(dop, "_shortable_cache", {}, raising=False)

    async def _drive():
        return [await dop._is_shortable("UNG") for _ in range(4)]

    assert all(asyncio.run(_drive()))
    assert calls["n"] == 1, f"fetched {calls['n']} times"
    dop._shortable_cache = {}


# ── buying that would flip a short ───────────────────────────────────────────
# Alpaca will not let ONE order flip a short into a long:
#   403 {"code":40310000,"message":"insufficient qty available for order
#        (requested: 1.77, available: 1)","symbol":"SPY"}
# Twice in a single run. Cap the buy at the short size so it CLOSES the short —
# the risk-reducing half of the intent, and what Alpaca requires be done first.


def _call_buy(monkeypatch, symbol, qty, held):
    monkeypatch.setattr(dop, "_position_map_cache", {symbol: held}, raising=False)
    out = asyncio.run(dop._equity_short_safe_qty(symbol, "buy", qty, False))
    dop._position_map_cache = None
    return out


def test_a_buy_that_would_flip_a_short_is_capped(monkeypatch):
    """THE BUG: SPY, short 1, buy 1.77 -> 403."""
    assert _call_buy(monkeypatch, "SPY", 1.77, held=-1.0) == 1.0


def test_a_buy_that_exactly_closes_a_short_is_untouched(monkeypatch):
    assert _call_buy(monkeypatch, "SPY", 1.0, held=-1.0) == 1.0


def test_a_partial_short_close_is_untouched(monkeypatch):
    """Buying less than the short just reduces it — legal."""
    assert _call_buy(monkeypatch, "SPY", 0.5, held=-1.0) == 0.5


def test_a_buy_with_no_short_is_untouched(monkeypatch):
    """The ordinary case: fractional longs are legal."""
    assert _call_buy(monkeypatch, "SPY", 1.77, held=0.0) == 1.77
    assert _call_buy(monkeypatch, "SPY", 1.77, held=10.0) == 1.77
