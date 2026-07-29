"""The 61-bot fleet has been evaluating on NO price data. Both paths failed.

Seen live in the Render logs on 2026-07-29, several times a minute:

    {"symbol": "SPY", "error": "'tuple' object has no attribute 'lower'",
     "event": "yfinance fallback failed"}
    {"symbol": "QQQ", "error": "'tuple' object has no attribute 'lower'", ...}

`_fetch_ohlcv` has two sources and BOTH were broken, so it always returned an
empty DataFrame:

1. REDIS — it read a hand-built key `ohlcv:{symbol}:1d`, while the writer
   (`price_cache.set_ohlcv`) uses `ohlcv:{exchange}:{symbol}:{interval}`. No
   exchange segment, so it missed on every symbol on every tick. This is the
   exact `prices:{symbol}` topic-vs-key class already documented in
   app/tasks/CLAUDE.md — a miss is indistinguishable from a cold cache, so it
   fell through silently to the fallback.

2. YFINANCE — `yf.download()` returns a MultiIndex even for a SINGLE ticker in
   yfinance >= 0.2.51: ('Close', 'SPY'). `[c.lower() for c in df.columns]`
   therefore calls .lower() on a tuple and raises.

With the fallback raising and the cache always missing, every bot evaluated
against `pd.DataFrame(columns=[...])`. The bots kept reporting
"Conditions not met" — indistinguishable, in the logs, from a genuine
no-signal verdict.

Installed yfinance here is 1.5.1, well past the version that introduced the
MultiIndex, so this was not a latent risk — it was firing continuously.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.bots import engine
from app import redis_client


# ── the yfinance MultiIndex ──────────────────────────────────────────────────

def _single_ticker_frame() -> pd.DataFrame:
    """What yfinance actually hands back for one ticker."""
    return pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 1_000]],
        columns=pd.MultiIndex.from_tuples(
            [("Open", "SPY"), ("High", "SPY"), ("Low", "SPY"),
             ("Close", "SPY"), ("Volume", "SPY")],
            names=["Price", "Ticker"],
        ),
    )


def test_the_old_lowercasing_really_did_raise():
    """Pins the reported error so the regression is unambiguous."""
    with pytest.raises(AttributeError, match="'tuple' object has no attribute 'lower'"):
        [c.lower() for c in _single_ticker_frame().columns]


@pytest.mark.asyncio
async def test_a_multiindex_download_yields_usable_columns(monkeypatch):
    monkeypatch.setattr(engine, "_map_crypto_symbol", lambda s: s, raising=False)

    class _YF:
        @staticmethod
        def download(*a, **k):
            return _single_ticker_frame()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", _YF)

    async def no_cache(*a, **k):
        return None
    monkeypatch.setattr(redis_client.price_cache, "get_ohlcv", no_cache, raising=False)

    df = await engine._fetch_ohlcv("SPY", "equity")
    assert "close" in df.columns, f"columns are {list(df.columns)}"
    assert not df.empty, "a valid download still produced an empty frame"
    assert float(df["close"].iloc[0]) == 100.5


@pytest.mark.asyncio
async def test_a_flat_column_download_still_works(monkeypatch):
    """Must not break if yfinance returns a plain Index."""
    class _YF:
        @staticmethod
        def download(*a, **k):
            return pd.DataFrame([[100.0, 100.5]], columns=["Open", "Close"])

    monkeypatch.setitem(__import__("sys").modules, "yfinance", _YF)

    async def no_cache(*a, **k):
        return None
    monkeypatch.setattr(redis_client.price_cache, "get_ohlcv", no_cache, raising=False)

    df = await engine._fetch_ohlcv("SPY", "equity")
    assert list(df.columns) == ["open", "close"]


# ── the Redis key ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_cache_is_read_through_the_shared_accessor(monkeypatch):
    """It must ask for the key the writer actually writes."""
    seen: list[tuple] = []

    async def spy_get_ohlcv(exchange, symbol, interval):
        seen.append((exchange, symbol, interval))
        return [{"close": 1.0} for _ in range(25)]

    monkeypatch.setattr(redis_client.price_cache, "get_ohlcv", spy_get_ohlcv, raising=False)

    df = await engine._fetch_ohlcv("SPY", "equity")
    assert seen == [("alpaca", "SPY", "1d")], (
        f"cache was queried as {seen} — the writer uses "
        f"ohlcv:{{exchange}}:{{symbol}}:{{interval}}"
    )
    assert len(df) == 25, "a cache HIT must short-circuit before yfinance"


@pytest.mark.asyncio
async def test_a_crypto_symbol_uses_the_crypto_namespace(monkeypatch):
    seen: list[tuple] = []

    async def spy_get_ohlcv(exchange, symbol, interval):
        seen.append((exchange, symbol, interval))
        return None

    monkeypatch.setattr(redis_client.price_cache, "get_ohlcv", spy_get_ohlcv, raising=False)

    class _YF:
        @staticmethod
        def download(*a, **k):
            return pd.DataFrame()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", _YF)
    await engine._fetch_ohlcv("BTC/USD", "crypto")
    assert seen and seen[0][0] == "crypto", seen


@pytest.mark.asyncio
async def test_a_short_cache_row_set_falls_through_to_yfinance(monkeypatch):
    """< 20 rows is not enough to evaluate — must not be used."""
    async def few(*a, **k):
        return [{"close": 1.0} for _ in range(5)]
    monkeypatch.setattr(redis_client.price_cache, "get_ohlcv", few, raising=False)

    called = {"n": 0}

    class _YF:
        @staticmethod
        def download(*a, **k):
            called["n"] += 1
            return pd.DataFrame([[1.0]], columns=["Close"])

    monkeypatch.setitem(__import__("sys").modules, "yfinance", _YF)
    await engine._fetch_ohlcv("SPY", "equity")
    assert called["n"] == 1, "a too-short cache hit was used instead of refetching"
