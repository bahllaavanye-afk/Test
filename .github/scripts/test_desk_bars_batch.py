"""Regression tests for the desk bar-fetch fix.

The crypto/equity desks placed 0 orders on every run because market-data bars
came back empty: every symbol's bars request fired concurrently against
Alpaca's free data tier, which 429'd nearly all of them (bars_fetched=2/12 →
signals_generated=0 → no trades). The fix batches all symbols into ONE request
per asset class and retries 429 with backoff.

These tests pin that behavior with a mocked Alpaca API (no network, no creds).
"""
from __future__ import annotations

import asyncio
import importlib.util
import urllib.error
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).parent / "desk_order_placer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("desk_order_placer_under_test", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


dop = _load_module()


def _bar(i: int) -> dict:
    return {"t": f"2026-01-{(i % 27) + 1:02d}T00:00:00Z",
            "o": 1.0 + i, "h": 2.0 + i, "l": 0.5 + i, "c": 1.5 + i, "v": 100 + i}


def test_batch_makes_one_request_per_asset_class():
    """8 symbols → exactly 2 upstream calls (1 crypto, 1 stock), not 8."""
    calls: list[tuple[str, str]] = []

    async def fake_get(path, params=None, data_api=False):
        calls.append((path, params["symbols"]))
        syms = params["symbols"].split(",")
        return {"bars": {s: [_bar(i) for i in range(60)] for s in syms},
                "next_page_token": None}

    dop._alpaca_get = fake_get
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "AAPL", "MSFT", "SPY"]
    out = asyncio.run(dop._get_bars_batch(symbols))

    assert len(calls) == 2, f"expected 2 batched calls, got {len(calls)}: {calls}"
    paths = {c[0] for c in calls}
    assert "/v1beta3/crypto/us/bars" in paths
    assert "/v2/stocks/bars" in paths
    # every symbol came back with a usable OHLCV frame
    assert set(out) == set(symbols)
    for sym in symbols:
        df = out[sym]
        assert len(df) == 60
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_batch_paginates_next_page_token():
    """A second page (next_page_token) is followed and concatenated."""
    pages = [
        {"bars": {"BTC/USD": [_bar(i) for i in range(30)]}, "next_page_token": "P2"},
        {"bars": {"BTC/USD": [_bar(i) for i in range(30, 60)]}, "next_page_token": None},
    ]
    seq = iter(pages)

    async def fake_get(path, params=None, data_api=False):
        return next(seq)

    dop._alpaca_get = fake_get
    out = asyncio.run(dop._get_bars_batch(["BTC/USD"]))
    assert len(out["BTC/USD"]) == 60  # both pages merged


def test_batch_handles_only_crypto_or_only_stocks():
    async def fake_get(path, params=None, data_api=False):
        syms = params["symbols"].split(",")
        return {"bars": {s: [_bar(i) for i in range(55)] for s in syms}, "next_page_token": None}

    dop._alpaca_get = fake_get
    crypto_only = asyncio.run(dop._get_bars_batch(["BTC/USD", "ETH/USD"]))
    assert set(crypto_only) == {"BTC/USD", "ETH/USD"}
    stocks_only = asyncio.run(dop._get_bars_batch(["AAPL", "MSFT"]))
    assert set(stocks_only) == {"AAPL", "MSFT"}


def test_alpaca_get_sync_retries_on_429(monkeypatch):
    """429 is retried with backoff, then succeeds — not surfaced as a failure."""
    attempts = {"n": 0}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok": true}'

    def fake_urlopen(req, timeout=8):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda *_: None)  # no real waiting in tests

    result = dop._alpaca_get_sync("/v2/account")
    assert result == {"ok": True}
    assert attempts["n"] == 3  # two 429s, third succeeds


def test_alpaca_get_sync_raises_after_persistent_429(monkeypatch):
    def fake_urlopen(req, timeout=8):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(urllib.error.HTTPError):
        dop._alpaca_get_sync("/v2/account")
