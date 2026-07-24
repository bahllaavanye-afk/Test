"""Crypto desk: drop delisted/non-tradable pairs before wasting a signal on them.

Fixes the class where a delisted pair (e.g. MKR/USD) still had bars, generated a
signal, and only failed at order time (422 'asset not active') every run. The
filter is fail-soft: any lookup problem keeps the full universe.
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_MOD = Path(__file__).parent / "desk_order_placer.py"
_spec = importlib.util.spec_from_file_location("dop_crypto_test", _MOD)
dop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dop)  # type: ignore[union-attr]

_UNIVERSE = ["BTC/USD", "ETH/USD", "UNI/USD", "MKR/USD"]


def test_drops_non_tradable_pair():
    tradable = {"BTC/USD", "ETH/USD", "UNI/USD"}          # MKR delisted
    kept, dropped = dop._filter_tradable_crypto(_UNIVERSE, tradable)
    assert dropped == ["MKR/USD"]
    assert kept == ["BTC/USD", "ETH/USD", "UNI/USD"]


def test_none_lookup_keeps_all():
    kept, dropped = dop._filter_tradable_crypto(_UNIVERSE, None)
    assert kept == _UNIVERSE and dropped == []


def test_format_mismatch_keeps_all():
    # tradable set is in the WRONG format (no 'BTC/USD') → never filter.
    kept, dropped = dop._filter_tradable_crypto(_UNIVERSE, {"BTCUSD", "ETHUSD"})
    assert kept == _UNIVERSE and dropped == []


def test_non_crypto_symbols_pass_through():
    equity = ["SPY", "QQQ", "AAPL"]
    kept, dropped = dop._filter_tradable_crypto(equity, {"BTC/USD", "ETH/USD"})
    assert kept == equity and dropped == []


def test_never_returns_empty_universe():
    # even if somehow nothing matches, never hand back an empty list.
    kept, dropped = dop._filter_tradable_crypto(["MKR/USD"], {"BTC/USD", "ETH/USD"})
    assert kept == ["MKR/USD"]  # kept-or-all fallback


def test_tradable_lookup_parses_and_caches(monkeypatch):
    dop._tradable_crypto_cache = None
    calls = {"n": 0}
    async def fake_get(path, params=None, data_api=False):
        calls["n"] += 1
        assert path == "/v2/assets" and params.get("asset_class") == "crypto"
        return [
            {"symbol": "BTC/USD", "tradable": True},
            {"symbol": "ETH/USD", "tradable": True},
            {"symbol": "DEAD/USD", "tradable": False},   # excluded
        ]
    monkeypatch.setattr(dop, "_alpaca_get", fake_get)
    out = asyncio.run(dop._tradable_crypto_symbols())
    assert out == {"BTC/USD", "ETH/USD"}
    # cached — second call does not hit the API again
    asyncio.run(dop._tradable_crypto_symbols())
    assert calls["n"] == 1


def test_tradable_lookup_failsoft_on_error(monkeypatch):
    dop._tradable_crypto_cache = None
    async def boom(path, params=None, data_api=False):
        raise RuntimeError("alpaca down")
    monkeypatch.setattr(dop, "_alpaca_get", boom)
    assert asyncio.run(dop._tradable_crypto_symbols()) is None
