"""FX desk no-repeat guard — the "same 3 orders every run" fix.

An open position in the SAME direction suppresses a new order for that pair;
opposite-direction signals stay allowed (they reduce/flip exposure). The guard
FAILS OPEN when the positions fetch errors — monitoring must never block the
desk. Mocks _oanda; no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fx_desk as fx


def _positions_payload():
    return {"positions": [
        {"instrument": "EUR_USD",
         "long": {"units": "3000"}, "short": {"units": "0"}},
        {"instrument": "GBP_USD",
         "long": {"units": "0"}, "short": {"units": "-2000"}},
        {"instrument": "USD_JPY",
         "long": {"units": "1000"}, "short": {"units": "-1000"}},  # net flat
    ]}


def test_open_positions_parses_net_units(monkeypatch):
    monkeypatch.setattr(fx, "OANDA_ACCOUNT_ID", "acct")
    monkeypatch.setattr(fx, "_oanda", lambda m, p, b=None: _positions_payload())
    pos = fx.open_positions()
    assert pos == {"EUR_USD": 3000.0, "GBP_USD": -2000.0}   # net-flat pair dropped


def test_same_direction_is_repeat():
    pos = {"EUR_USD": 3000.0, "GBP_USD": -2000.0}
    assert fx.is_repeat("EUR_USD", "buy", pos)      # already long
    assert fx.is_repeat("GBP_USD", "sell", pos)     # already short


def test_opposite_direction_is_allowed():
    pos = {"EUR_USD": 3000.0, "GBP_USD": -2000.0}
    assert not fx.is_repeat("EUR_USD", "sell", pos)  # reduces/flips → allowed
    assert not fx.is_repeat("GBP_USD", "buy", pos)


def test_unpositioned_pair_is_allowed():
    assert not fx.is_repeat("USD_JPY", "buy", {"EUR_USD": 3000.0})


def test_fetch_failure_fails_open(monkeypatch):
    monkeypatch.setattr(fx, "OANDA_ACCOUNT_ID", "acct")
    def boom(m, p, b=None):
        raise RuntimeError("api down")
    monkeypatch.setattr(fx, "_oanda", boom)
    assert fx.open_positions() is None
    assert not fx.is_repeat("EUR_USD", "buy", None)  # guard off, desk trades
