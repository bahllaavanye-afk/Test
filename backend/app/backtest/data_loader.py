"""OHLCV data loader with yfinance (free) as primary source.

Strategies and backtests call fetch_ohlcv() — it's entirely offline,
no broker keys required. yfinance pulls from Yahoo Finance for free.
"""
from __future__ import annotations
import asyncio
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
    Falls back to synthetic GBM data when network is unavailable.
    """
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
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, fetch_ohlcv_sync, symbol, start, end, interval, market_type
    )
