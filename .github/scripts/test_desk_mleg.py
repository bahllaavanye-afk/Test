"""Options-desk real multi-leg routing: income signals place actual spreads.

Pins: structure leg specs, nearest-strike/expiry contract picking, the mleg
payload Alpaca requires, and the cardinal rule — an unresolvable leg places
NOTHING (fall back to underlying proxy), never a partial spread.
"""
from __future__ import annotations

import asyncio
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_MOD = Path(__file__).parent / "desk_order_placer.py"


def _load():
    spec = importlib.util.spec_from_file_location("dop_mleg_test", _MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


dop = _load()


# ── leg specs ────────────────────────────────────────────────────────────────

def test_credit_spread_is_directional():
    bull = dop._income_leg_spec("credit_spread_income", "buy")
    assert [(t, s) for t, s, _ in bull] == [("put", "sell"), ("put", "buy")]
    bear = dop._income_leg_spec("credit_spread_income", "sell")
    assert [(t, s) for t, s, _ in bear] == [("call", "sell"), ("call", "buy")]


def test_iron_condor_is_four_leg_any_side():
    for side in ("buy", "sell"):
        legs = dop._income_leg_spec("iron_condor", side)
        assert len(legs) == 4
        assert sum(1 for _, s, _ in legs if s == "sell") == 2


def test_non_income_strategy_returns_none():
    assert dop._income_leg_spec("momentum", "buy") is None
    assert dop._income_leg_spec("cash_secured_put", "sell") is None   # long-only structure


# ── contract picking ─────────────────────────────────────────────────────────

def _c(strike, exp, sym=None):
    return {"strike_price": str(strike), "expiration_date": exp,
            "symbol": sym or f"SPY{exp.replace('-', '')[2:]}P{int(strike*1000):08d}"}


def test_pick_contract_nearest_strike_then_expiry():
    target_exp = (datetime.now(timezone.utc).date() + timedelta(days=35)).isoformat()
    near = (datetime.now(timezone.utc).date() + timedelta(days=33)).isoformat()
    far = (datetime.now(timezone.utc).date() + timedelta(days=49)).isoformat()
    picked = dop._pick_contract([_c(580, far), _c(590, near), _c(590, far)], 590, target_exp)
    assert float(picked["strike_price"]) == 590 and picked["expiration_date"] == near


def test_pick_contract_skips_malformed_and_empty():
    assert dop._pick_contract([], 590, "2026-08-20") is None
    assert dop._pick_contract([{"strike_price": "bad"}], 590, "2026-08-20") is None


# ── placement ────────────────────────────────────────────────────────────────

def _contracts_response(opt_type, spot):
    exp = (datetime.now(timezone.utc).date() + timedelta(days=35)).isoformat()
    strikes = ([spot * m for m in (0.96, 0.92, 0.95, 0.90, 0.91)] if opt_type == "put"
               else [spot * m for m in (1.04, 1.08, 1.05, 1.09)])
    return {"option_contracts": [_c(round(k), exp) for k in strikes]}


def test_place_income_spread_builds_mleg_payload(monkeypatch):
    posted = {}
    async def fake_get(path, params=None, data_api=False):
        assert "/v2/options/contracts" in path
        return _contracts_response(params["type"], 600.0)
    async def fake_post(path, body):
        posted["path"], posted["body"] = path, body
        return {"id": "mleg-9", "status": "accepted", "legs": body["legs"]}
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    monkeypatch.setattr(dop, "_alpaca_post", fake_post)

    out = asyncio.run(dop._place_income_spread("SPY", "credit_spread_income", "buy", 600.0))
    assert out and out["id"] == "mleg-9"
    b = posted["body"]
    assert b["order_class"] == "mleg" and b["qty"] == "1" and posted["path"] == "/v2/orders"
    assert len(b["legs"]) == 2
    assert b["legs"][0]["position_intent"] == "sell_to_open"
    assert b["legs"][1]["position_intent"] == "buy_to_open"
    # short strike above long strike for a bull put spread
    assert float(b["legs"][0]["symbol"][-8:]) > float(b["legs"][1]["symbol"][-8:])


def test_unresolvable_leg_places_nothing(monkeypatch):
    async def fake_get(path, params=None, data_api=False):
        return {"option_contracts": []}                # no contracts at all
    async def must_not_post(path, body):
        raise AssertionError("partial spread posted!")
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    monkeypatch.setattr(dop, "_alpaca_post", must_not_post)
    assert asyncio.run(dop._place_income_spread("SPY", "iron_condor", "buy", 600.0)) is None


def test_contracts_fetch_error_places_nothing(monkeypatch):
    async def fake_get(path, params=None, data_api=False):
        raise RuntimeError("403 options not enabled")
    async def must_not_post(path, body):
        raise AssertionError("posted despite fetch error!")
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    monkeypatch.setattr(dop, "_alpaca_post", must_not_post)
    assert asyncio.run(dop._place_income_spread("SPY", "wheel", "buy", 600.0)) is None


def test_bad_spot_places_nothing():
    assert asyncio.run(dop._place_income_spread("SPY", "iron_condor", "buy", 0.0)) is None
