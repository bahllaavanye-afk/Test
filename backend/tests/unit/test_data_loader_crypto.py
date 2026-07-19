"""Tests for the Alpaca crypto data path in data_loader.

Binance is geo-blocked (451), so crypto OHLCV now comes from Alpaca's free public
bars API with yfinance → synthetic as fallback. These tests mock the HTTP boundary
(`_http_get_json`) so they never touch the network.
"""
from __future__ import annotations

import sys
from datetime import date
from typing import Callable, List, Dict

import pandas as pd
import pytest

import app.backtest.data_loader as dl


def _page(bars: List[dict], token: str | None = None) -> dict:
    """Create a mock Alpaca pagination response."""
    return {"bars": {"BTC/USD": bars}, "next_page_token": token}


def _raise(*_a, **_k):
    """Simulate an Alpaca failure."""
    raise RuntimeError("simulated Alpaca failure")


def _make_fake_http_get(pages: List[dict]) -> Callable[[str, dict, float], dict]:
    """Return a callable mimicking ``_http_get_json`` that yields the supplied pages.

    The returned function keeps internal state to serve pages sequentially each
    time it is invoked, mirroring Alpaca's pagination behaviour.
    """
    state = {"index": 0}

    def _fake_get(url: str, headers: dict, timeout: float = 20.0) -> dict:
        page = pages[state["index"]]
        state["index"] += 1
        return page

    return _fake_get


def test_symbol_normalization():
    assert dl._symbol_to_alpaca_crypto("BTC/USDT") == "BTC/USD"
    assert dl._symbol_to_alpaca_crypto("ETH-USD") == "ETH/USD"
    assert dl._symbol_to_alpaca_crypto("SOLUSDT") == "SOL/USD"
    assert dl._symbol_to_alpaca_crypto("btc") == "BTC/USD"


def test_interval_mapping():
    assert dl._interval_to_alpaca("1d") == "1Day"
    assert dl._interval_to_alpaca("1h") == "1Hour"
    assert dl._interval_to_alpaca("4h") == "4Hour"
    assert dl._interval_to_alpaca("totally-unknown") == "1Day"  # safe default


def test_fetch_alpaca_crypto_paginates_and_parses(monkeypatch):
    pages = [
        _page(
            [
                {
                    "t": "2024-01-01T00:00:00Z",
                    "o": 1.0,
                    "h": 2.0,
                    "l": 0.5,
                    "c": 1.5,
                    "v": 10,
                }
            ],
            token="tok2",
        ),
        _page(
            [
                {
                    "t": "2024-01-02T00:00:00Z",
                    "o": 1.5,
                    "h": 2.5,
                    "l": 1.0,
                    "c": 2.0,
                    "v": 12,
                }
            ],
            token=None,
        ),
    ]

    monkeypatch.setattr(dl, "_http_get_json", _make_fake_http_get(pages))

    df = dl._fetch_alpaca_crypto("BTC/USDT", date(2024, 1, 1), date(2024, 1, 2), "1d")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    # ``_make_fake_http_get`` should have consumed both pages.
    assert df["close"].tolist() == [1.5, 2.0]
    assert df.index.tz is None, "index must be tz-naive"
    assert df.index.is_monotonic_increasing


def test_fetch_ohlcv_sync_routes_crypto_to_alpaca(monkeypatch):
    monkeypatch.setattr(
        dl,
        "_http_get_json",
        lambda url, headers, timeout=20.0: _page(
            [
                {
                    "t": "2024-01-01T00:00:00Z",
                    "o": 1.0,
                    "h": 2.0,
                    "l": 0.5,
                    "c": 1.5,
                    "v": 10,
                }
            ]
        ),
    )
    df = dl.fetch_ohlcv_sync(
        "BTC/USDT", date(2024, 1, 1), date(2024, 1, 2), "1d", market_type="crypto"
    )
    assert len(df) == 1
    assert float(df["close"].iloc[0]) == 1.5


def test_crypto_falls_back_when_alpaca_fails(monkeypatch):
    # Alpaca errors AND yfinance is unavailable → must fall back to synthetic, not crash.
    monkeypatch.setattr(dl, "_http_get_json", _raise)
    monkeypatch.setitem(sys.modules, "yfinance", None)  # `import yfinance` → ImportError
    df = dl.fetch_ohlcv_sync(
        "BTC/USDT", date(2024, 1, 1), date(2024, 3, 1), "1d", market_type="crypto"
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty  # synthetic GBM series
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_end_bound_is_end_of_day_not_next_day(monkeypatch):
    # Alpaca's `end` is inclusive, so we must bound at end-of-day of `end`,
    # never request the day after `end` (that pulled an extra bar).
    import urllib.parse as up

    captured: dict[str, str] = {}

    def fake_get(url: str, headers: dict, timeout: float = 20.0) -> dict:
        captured["url"] = url
        return _page(
            [
                {
                    "t": "2024-01-02T00:00:00Z",
                    "o": 1.0,
                    "h": 2.0,
                    "l": 0.5,
                    "c": 1.5,
                    "v": 10,
                }
            ]
        )

    monkeypatch.setattr(dl, "_http_get_json", fake_get)
    dl._fetch_alpaca_crypto("BTC/USDT", date(2024, 1, 1), date(2024, 1, 2), "1d")

    q = up.parse_qs(up.urlparse(captured["url"]).query)
    assert q["start"][0] == "2024-01-01"
    assert q["end"][0] == "2024-01-02T23:59:59Z"
    assert "2024-01-03" not in captured["url"]  # never request the day after `end`


def test_empty_response_returns_empty_df(monkeypatch):
    monkeypatch.setattr(
        dl,
        "_http_get_json",
        lambda url, headers, timeout=20.0: {"bars": {}, "next_page_token": None},
    )
    df = dl._fetch_alpaca_crypto("BTC/USDT", date(2024, 1, 1), date(2024, 1, 2), "1d")
    assert df.empty


def test_equity_path_unaffected(monkeypatch):
    # Non-crypto must never hit the Alpaca crypto endpoint.
    monkeypatch.setattr(dl, "_http_get_json", _raise)
    monkeypatch.setitem(sys.modules, "yfinance", None)
    df = dl.fetch_ohlcv_sync(
        "AAPL", date(2024, 1, 1), date(2024, 2, 1), "1d", market_type="equity"
    )
    assert isinstance(df, pd.DataFrame)  # synthetic; crypto branch skipped