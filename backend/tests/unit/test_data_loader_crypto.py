"""Tests for the Alpaca crypto data path in data_loader.

Binance is geo-blocked (451), so crypto OHLCV now comes from Alpaca's free public
bars API with yfinance → synthetic as fallback. These tests mock the HTTP boundary
(`_http_get_json`) so they never touch the network.
"""
from __future__ import annotations

import sys
from datetime import date

import pandas as pd
import pytest

import app.backtest.data_loader as dl

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_TIMEOUT = 20.0
DEFAULT_INTERVAL_FALLBACK = "1Day"

SAMPLE_BAR_1 = {"t": "2024-01-01T00:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 10}
SAMPLE_BAR_2 = {"t": "2024-01-02T00:00:00Z", "o": 1.5, "h": 2.5, "l": 1.0, "c": 2.0, "v": 12}

EXPECTED_COLUMNS = ["open", "high", "low", "close", "volume"]

START_DATE_STR = "2024-01-01"
END_DATE_STR = "2024-01-02T23:59:59Z"
NEXT_DAY_STR = "2024-01-03"

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _page(bars: list[dict], token=None) -> dict:
    return {"bars": {"BTC/USD": bars}, "next_page_token": token}


def _raise(*_a, **_k):
    raise RuntimeError("simulated Alpaca failure")


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_symbol_normalization():
    assert dl._symbol_to_alpaca_crypto("BTC/USDT") == "BTC/USD"
    assert dl._symbol_to_alpaca_crypto("ETH-USD") == "ETH/USD"
    assert dl._symbol_to_alpaca_crypto("SOLUSDT") == "SOL/USD"
    assert dl._symbol_to_alpaca_crypto("btc") == "BTC/USD"


def test_interval_mapping():
    assert dl._interval_to_alpaca("1d") == "1Day"
    assert dl._interval_to_alpaca("1h") == "1Hour"
    assert dl._interval_to_alpaca("4h") == "4Hour"
    assert dl._interval_to_alpaca("totally-unknown") == DEFAULT_INTERVAL_FALLBACK  # safe default


def test_fetch_alpaca_crypto_paginates_and_parses(monkeypatch):
    pages = [
        _page([SAMPLE_BAR_1], token="tok2"),
        _page([SAMPLE_BAR_2], token=None),
    ]
    seen = {"i": 0}

    def fake_get(url, headers, timeout=DEFAULT_TIMEOUT):
        page = pages[seen["i"]]
        seen["i"] += 1
        return page

    monkeypatch.setattr(dl, "_http_get_json", fake_get)
    df = dl._fetch_alpaca_crypto("BTC/USDT", date(2024, 1, 1), date(2024, 1, 2), "1d")

    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 2
    assert seen["i"] == 2, "should have followed next_page_token"
    assert df["close"].tolist() == [1.5, 2.0]
    assert df.index.tz is None, "index must be tz-naive"
    assert df.index.is_monotonic_increasing


def test_fetch_ohlcv_sync_routes_crypto_to_alpaca(monkeypatch):
    monkeypatch.setattr(
        dl,
        "_http_get_json",
        lambda url, headers, timeout=DEFAULT_TIMEOUT: _page([SAMPLE_BAR_1]),
    )
    df = dl.fetch_ohlcv_sync("BTC/USDT", date(2024, 1, 1), date(2024, 1, 2), "1d",
                             market_type="crypto")
    assert len(df) == 1
    assert float(df["close"].iloc[0]) == 1.5


def test_crypto_falls_back_when_alpaca_fails(monkeypatch):
    # Alpaca errors AND yfinance is unavailable → must fall back to synthetic, not crash.
    monkeypatch.setattr(dl, "_http_get_json", _raise)
    monkeypatch.setitem(sys.modules, "yfinance", None)  # `import yfinance` → ImportError
    df = dl.fetch_ohlcv_sync("BTC/USDT", date(2024, 1, 1), date(2024, 3, 1), "1d",
                             market_type="crypto")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty  # synthetic GBM series
    assert list(df.columns) == EXPECTED_COLUMNS


def test_end_bound_is_end_of_day_not_next_day(monkeypatch):
    # Alpaca's `end` is inclusive, so we must bound at end-of-day of `end`,
    # never request the day after `end` (that pulled an extra bar).
    import urllib.parse as up

    captured: dict[str, str] = {}

    def fake_get(url, headers, timeout=DEFAULT_TIMEOUT):
        captured["url"] = url
        return _page([SAMPLE_BAR_2])

    monkeypatch.setattr(dl, "_http_get_json", fake_get)
    dl._fetch_alpaca_crypto("BTC/USDT", date(2024, 1, 1), date(2024, 1, 2), "1d")

    q = up.parse_qs(up.urlparse(captured["url"]).query)
    assert q["start"][0] == START_DATE_STR
    assert q["end"][0] == END_DATE_STR
    assert NEXT_DAY_STR not in captured["url"]  # never request the day after `end`


def test_empty_response_returns_empty_df(monkeypatch):
    monkeypatch.setattr(
        dl,
        "_http_get_json",
        lambda url, headers, timeout=DEFAULT_TIMEOUT: {"bars": {}, "next_page_token": None},
    )
    df = dl._fetch_alpaca_crypto("BTC/USDT", date(2024, 1, 1), date(2024, 1, 2), "1d")
    assert df.empty


def test_equity_path_unaffected(monkeypatch):
    # Non-crypto must never hit the Alpaca crypto endpoint.
    monkeypatch.setattr(dl, "_http_get_json", _raise)
    monkeypatch.setitem(sys.modules, "yfinance", None)
    df = dl.fetch_ohlcv_sync("AAPL", date(2024, 1, 1), date(2024, 2, 1), "1d",
                             market_type="equity")
    assert isinstance(df, pd.DataFrame)  # synthetic; crypto branch skipped