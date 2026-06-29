"""OHLCV data loader with yfinance (free) as primary source.

Strategies and backtests call fetch_ohlcv() — it's entirely offline,
no broker keys required. yfinance pulls from Yahoo Finance for free.
"""
from __future__ import annotations
import asyncio
import os
import pandas as pd
from datetime import date, timedelta
from app.utils.logging import logger


def _interval_to_yf(interval: str) -> str:
    """Convert internal interval names to yfinance format."""
    _MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h",
        "1d": "1d", "1wk": "1wk", "1mo": "1mo",
        "daily": "1d", "hourly": "1h", "weekly": "1wk",
    }
    return _MAP.get(interval.lower(), "1d")


def _symbol_to_yf(symbol: str, market_type: str = "equity") -> str:
    """Convert internal symbol format to yfinance ticker."""
    if market_type == "crypto":
        # BTC/USDT → BTC-USD; ETH/USDT → ETH-USD
        base = symbol.replace("/USDT", "").replace("/USD", "").replace("/BTC", "")
        return f"{base}-USD"
    return symbol.upper()


# ── Alpaca crypto market data (free, keyless public endpoint) ──────────────────
# Binance is geo-blocked (HTTP 451) in our deploy region, so crypto OHLCV came
# from rate-limited yfinance (→ synthetic). Alpaca's crypto bars API is public
# (no key required), not geo-blocked, and returns real exchange data — so it's the
# primary crypto source, with yfinance → synthetic kept as fallback.
_ALPACA_CRYPTO_BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"

_INTERVAL_TO_ALPACA = {
    "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
    "1h": "1Hour", "2h": "2Hour", "4h": "4Hour",
    "1d": "1Day", "1wk": "1Week", "1mo": "1Month",
    "daily": "1Day", "hourly": "1Hour", "weekly": "1Week",
}


def _interval_to_alpaca(interval: str) -> str:
    return _INTERVAL_TO_ALPACA.get(interval.lower(), "1Day")


def _symbol_to_alpaca_crypto(symbol: str) -> str:
    """Normalize an internal crypto symbol to Alpaca's `BASE/USD` pair format.

    Handles BTC/USDT, BTC-USD, BTCUSDT, BTC → all become BTC/USD.
    """
    s = symbol.upper().replace("-", "/").strip()
    if "/" in s:
        base = s.split("/")[0]
    else:
        base = s
        for quote in ("USDT", "USDC", "USD"):
            if s.endswith(quote) and len(s) > len(quote):
                base = s[: -len(quote)]
                break
    return f"{base}/USD"


def _http_get_json(url: str, headers: dict, timeout: float = 20.0, retries: int = 1) -> dict:
    """Minimal stdlib JSON GET with a light retry (kept tiny + patchable for tests)."""
    import json
    import time
    import urllib.request

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed host)
                return json.loads(resp.read().decode())
        except Exception as exc:  # transient network/5xx — back off briefly, then retry
            last_exc = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _fetch_alpaca_crypto(
    symbol: str, start: date, end: date, interval: str, max_pages: int = 25
) -> pd.DataFrame:
    """Fetch crypto OHLCV from Alpaca's public crypto bars API.

    Returns a tz-naive DataFrame [open, high, low, close, volume] sorted ascending,
    or an empty DataFrame if no bars are returned. Follows next_page_token.
    """
    import urllib.parse

    pair = _symbol_to_alpaca_crypto(symbol)
    timeframe = _interval_to_alpaca(interval)
    headers = {"User-Agent": "QuantEdge/1.0", "Accept": "application/json"}
    # Crypto bars are public, but sending keys (when present) raises rate limits.
    key = os.environ.get("ALPACA_API_KEY", "")
    sec = os.environ.get("ALPACA_SECRET_KEY", "")
    if key and sec and key != "test-key":
        headers["APCA-API-KEY-ID"] = key
        headers["APCA-API-SECRET-KEY"] = sec

    rows: list[dict] = []
    page_token: str | None = None
    for _ in range(max_pages):
        params = {
            "symbols": pair,
            "timeframe": timeframe,
            "start": start.isoformat(),
            # Alpaca's `end` is INCLUSIVE of bar timestamps. Use end-of-day so we
            # capture every bar on `end` (the 00:00 daily bar *and* all intraday
            # bars) without pulling the next day's bar. (yfinance needs +1 day;
            # Alpaca does not — copying that idiom here pulled one extra bar.)
            "end": f"{end.isoformat()}T23:59:59Z",
            "limit": 10000,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        url = f"{_ALPACA_CRYPTO_BARS_URL}?{urllib.parse.urlencode(params)}"
        payload = _http_get_json(url, headers, timeout=20.0)
        rows.extend((payload.get("bars") or {}).get(pair, []))
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    else:
        # max_pages exhausted without consuming the last page → bars were truncated
        if page_token:
            logger.warning(
                f"Alpaca crypto: hit max_pages={max_pages} for {pair} ({interval}); "
                "older bars may be truncated — widen max_pages or narrow the range"
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "ts"}
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    df = (
        df.set_index("ts")[["open", "high", "low", "close", "volume"]]
        .astype(float)
        .sort_index()
    )
    df = df[~df.index.duplicated(keep="last")]
    return df


def _synthetic_ohlcv(symbol: str, start: date, end: date, interval: str) -> pd.DataFrame:
    """
    Generate synthetic OHLCV using Geometric Brownian Motion when live data is
    unavailable (no network, delisted ticker, etc.).

    Deterministic seed based on symbol so results are reproducible.
    Returns realistic-looking daily bars with drift ≈ 10% pa, vol ≈ 15% pa.
    """
    import numpy as np

    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    if n < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(sum(ord(c) for c in symbol))
    mu = 0.10 / 252    # 10% annual drift
    sigma = 0.15 / 252 ** 0.5
    log_returns = rng.normal(mu - 0.5 * sigma ** 2, sigma, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    noise = rng.uniform(0.998, 1.002, n)
    open_ = np.roll(close, 1) * noise
    open_[0] = close[0] * 0.999
    high = np.maximum(open_, close) * rng.uniform(1.000, 1.010, n)
    low  = np.minimum(open_, close) * rng.uniform(0.990, 1.000, n)
    volume = rng.integers(1_000_000, 50_000_000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates),
    )
    logger.info(f"Synthetic OHLCV: {len(df)} bars for {symbol} (no live data available)")
    return df


def fetch_ohlcv_sync(
    symbol: str,
    start: date,
    end: date,
    interval: str = "1d",
    market_type: str = "equity",
) -> pd.DataFrame:
    """
    Fetch OHLCV data synchronously via yfinance (free, no API key needed).
    Returns DataFrame with columns: open, high, low, close, volume (lowercase).
    Crypto routes to Alpaca's free public bars API first; everything falls back to
    yfinance, then to synthetic GBM data when the network is unavailable.
    """
    if market_type == "crypto":
        try:
            df = _fetch_alpaca_crypto(symbol, start, end, interval)
            if not df.empty:
                logger.info(f"Alpaca crypto: loaded {len(df)} bars for {symbol} ({interval})")
                return df
            logger.warning(f"Alpaca crypto returned no bars for {symbol} — trying yfinance")
        except Exception as exc:
            logger.warning(f"Alpaca crypto fetch failed for {symbol}: {exc} — trying yfinance")

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — using synthetic data")
        return _synthetic_ohlcv(symbol, start, end, interval)

    yf_symbol = _symbol_to_yf(symbol, market_type)
    yf_interval = _interval_to_yf(interval)

    try:
        ticker = yf.Ticker(yf_symbol)
        # Add 1 day buffer on end so end date is inclusive
        end_buf = end + timedelta(days=1)
        df = ticker.history(
            start=start.isoformat(),
            end=end_buf.isoformat(),
            interval=yf_interval,
            auto_adjust=True,
        )
        if df.empty:
            logger.warning(f"yfinance returned no data for {yf_symbol} — using synthetic")
            return _synthetic_ohlcv(symbol, start, end, interval)

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"stock splits": "stock_splits", "capital gains": "capital_gains"})
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df = df.dropna()
        logger.info(f"yfinance: loaded {len(df)} bars for {yf_symbol} ({interval})")
        return df
    except Exception as exc:
        logger.warning(f"yfinance fetch failed for {yf_symbol}: {exc} — using synthetic")
        return _synthetic_ohlcv(symbol, start, end, interval)


async def fetch_ohlcv(
    symbol: str,
    start: date,
    end: date,
    interval: str = "1d",
    market_type: str = "equity",
) -> pd.DataFrame:
    """Async wrapper — runs the sync yfinance call in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, fetch_ohlcv_sync, symbol, start, end, interval, market_type
    )
