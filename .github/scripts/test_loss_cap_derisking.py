"""Loss-cap de-risking exemption — the fix for Monday 2026-07-20's frozen book.

Evidence: every market-hours desk run logged "🛑 DAILY LOSS CAP: equity down
2.72% vs prior close — no new orders this run" (weekend crypto drift vs
Friday's last_equity), blocking ALL orders including exits. Under the cap,
risk-REDUCING orders must stay allowed; only exposure-increasing orders are
blocked. Position fetch failure → {} → everything counts as non-reducing
(cap stays strict, the safe pre-fix behavior).
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_MOD = Path(__file__).parent / "desk_order_placer.py"
_spec = importlib.util.spec_from_file_location("dop_cap_test", _MOD)
dop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dop)  # type: ignore[union-attr]


def test_cap_still_trips_on_drawdown():
    assert dop.daily_loss_cap_hit(equity=97.0, last_equity=100.0, cap=0.02)
    assert not dop.daily_loss_cap_hit(equity=99.0, last_equity=100.0, cap=0.02)


def test_sell_against_long_is_reducing():
    assert dop.is_risk_reducing("sell", position_qty=10.0)


def test_buy_against_short_is_reducing():
    assert dop.is_risk_reducing("buy", position_qty=-5.0)


def test_new_exposure_is_not_reducing():
    assert not dop.is_risk_reducing("buy", position_qty=10.0)    # pyramiding a long
    assert not dop.is_risk_reducing("sell", position_qty=-5.0)   # adding to a short
    assert not dop.is_risk_reducing("buy", position_qty=0.0)     # fresh long
    assert not dop.is_risk_reducing("sell", position_qty=0.0)    # fresh short


def test_position_map_parses_signed_qty(monkeypatch):
    async def fake_get(path, params=None, data_api=False):
        assert path == "/v2/positions"
        return [
            {"symbol": "SPY", "qty": "12", "side": "long"},
            {"symbol": "QQQ", "qty": "7", "side": "short"},
            {"symbol": "BTCUSD", "qty": "0.5", "side": "long"},
            {"symbol": "ZERO", "qty": "0", "side": "long"},   # dropped
        ]
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    out = asyncio.run(dop._alpaca_position_map())
    assert out == {"SPY": 12.0, "QQQ": -7.0, "BTCUSD": 0.5}


def test_position_map_fetch_failure_is_empty(monkeypatch):
    async def boom(path, params=None, data_api=False):
        raise RuntimeError("api down")
    monkeypatch.setattr(dop, "_alpaca_get", boom)
    assert asyncio.run(dop._alpaca_position_map()) == {}
