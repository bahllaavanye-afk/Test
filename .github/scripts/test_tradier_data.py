"""Tradier options adapter — pure parsing + delta-picking guards (no network).

Mocks tradier_data._get so the tests never hit the API. Verifies expiration
sorting, nearest-DTE selection, delta-based strike picking, ATM IV, and that
every path fails soft (returns None/[] rather than raising) when the feed is down.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tradier_data as td


def _exp(days: int) -> str:
    return (date.today() + timedelta(days=days)).strftime("%Y-%m-%d")


def test_nearest_expiration_picks_closest_dte(monkeypatch):
    monkeypatch.setattr(td, "TOKEN", "x")
    monkeypatch.setattr(td, "_get", lambda p, timeout=20: {
        "expirations": {"date": [_exp(7), _exp(35), _exp(60)]}})
    assert td.nearest_expiration("SPY", 30) == _exp(35)
    assert td.nearest_expiration("SPY", 5) == _exp(7)


def test_pick_by_delta_selects_nearest_delta(monkeypatch):
    monkeypatch.setattr(td, "TOKEN", "x")

    def fake_get(path, timeout=20):
        if "expirations" in path:
            return {"expirations": {"date": [_exp(35)]}}
        # a chain of puts at increasing distance from ATM
        return {"options": {"option": [
            {"symbol": "P740", "strike": 740, "option_type": "put", "greeks": {"delta": -0.45}},
            {"symbol": "P728", "strike": 728, "option_type": "put", "greeks": {"delta": -0.30}},
            {"symbol": "P710", "strike": 710, "option_type": "put", "greeks": {"delta": -0.16}},
            {"symbol": "C760", "strike": 760, "option_type": "call", "greeks": {"delta": 0.30}},
        ]}}

    monkeypatch.setattr(td, "_get", fake_get)
    pick = td.pick_by_delta("SPY", 35, 0.30, "put")
    assert pick is not None and pick["symbol"] == "P728"      # closest to 0.30 on the put side
    # right-side filter: a 0.30 call must come from the call, not the -0.30 put
    call = td.pick_by_delta("SPY", 35, 0.30, "call")
    assert call is not None and call["option_type"] == "call" and call["symbol"] == "C760"


def test_atm_iv_uses_strike_nearest_spot(monkeypatch):
    monkeypatch.setattr(td, "TOKEN", "x")

    def fake_get(path, timeout=20):
        if "quotes" in path:
            return {"quotes": {"quote": {"last": 743.0, "bid": 742, "ask": 744}}}
        if "expirations" in path:
            return {"expirations": {"date": [_exp(30)]}}
        return {"options": {"option": [
            {"strike": 700, "option_type": "call", "greeks": {"mid_iv": 0.22}},
            {"strike": 745, "option_type": "call", "greeks": {"mid_iv": 0.15}},   # nearest 743
            {"strike": 800, "option_type": "call", "greeks": {"mid_iv": 0.19}},
        ]}}

    monkeypatch.setattr(td, "_get", fake_get)
    assert td.atm_iv("SPY", 30) == 0.15


def test_all_paths_fail_soft_when_feed_down(monkeypatch):
    monkeypatch.setattr(td, "TOKEN", "x")
    monkeypatch.setattr(td, "_get", lambda p, timeout=20: None)   # every call "fails"
    assert td.quote("SPY") is None
    assert td.expirations("SPY") == []
    assert td.nearest_expiration("SPY", 30) is None
    assert td.chain("SPY", _exp(30)) == []
    assert td.pick_by_delta("SPY", 30, 0.3, "put") is None
    assert td.atm_iv("SPY") is None


def test_no_token_means_unavailable(monkeypatch):
    monkeypatch.setattr(td, "TOKEN", "")
    assert td.available() is False
    assert td._get("/anything") is None      # short-circuits without a token
