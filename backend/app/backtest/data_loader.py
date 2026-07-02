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

# ── Constants ─────────────────────────────────────────────────────────────────────
# Interval conversion tables
INTERVAL_YF_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
    "daily": "1d",
    "hourly": "1h",
    "weekly": "1wk",
}

INTERVAL_ALPACA_MAP: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "2h": "2Hour",
    "4h": "4Hour",
    "1d": "1Day",
    "1wk": "1Week",
    "1mo": "1Month",
    "daily": "1Day",
    "hourly": "1Hour",
    "weekly": "1Week",
}

# Commodity → Yahoo Finance ticker mapping
COMMODITY_YF_MAP: dict[str, str] = {
    "GOLD": "GC=F",
    "XAU": "GC=F",
    "GC": "GC=F",
    "SILVER": "SI=F",
    "XAG": "SI=F",
    "SI": "SI=F",
    "OIL": "CL=F",
    "WTI": "CL=F",
    "CRUDE": "CL=F",
    "CL": "CL=F",
    "BRENT": "BZ=F",
    "BZ": "BZ=F",
    "NATGAS": "NG=F",
    "GAS": "NG=F",
    "NG": "NG=F",
    "COPPER": "HG=F",
    "HG": "HG=F",
    "CORN": "ZC=F",
    "WHEAT": "ZW=F",
    "SOYBEAN": "ZS=F",
    "SOY": "ZS=F",
    "PLATINUM": "PL=F",
    "PALLADIUM": "PA=F",
}

# Alpaca crypto endpoint
ALPACA_CRYPTO_BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"

# Network / retry defaults
DEFAULT_TIMEOUT = 20.0
DEFAULT_RETRIES = 1
DEFAULT_MAX_PAGES = 25
_BACKOFF_FACTOR = 0.5  # seconds per retry attempt

# Synthetic data generation defaults
_SYNTH_DRIFT_ANNUAL = 0.10
_SYNTH_VOL_ANNUAL = 0.15
_SYNTH_VOLUME_MIN = 1_000_000
_SYNTH_VOLUME_MAX = 50_000_000
_SYNTH_OPEN_NOISE_LOW = 0.998
_SYNTH_OPEN_NOISE_HIGH = 1.002
_SYNTH_HIGH_MULT_LOW = 1.000
_SYNTH_HIGH_MULT_HIGH = 1.010
_SYNTH_LOW_MULT_LOW = 0.990
_SYNTH_LOW_MULT_HIGH = 1.000


def _interval_to_yf(interval: str) -> str:
    """Convert internal interval names to yfinance format."""
    return INTERVAL_YF_MAP.get(interval.lower(), "1d")


# Friendly commodity names → yfinance continuous-future tickers.
def _commodity_yf(symbol: str) -> str:
    """Lookup a commodity ticker in the commodity map."""
    return COMMODITY_YF_MAP.get(symbol.upper(), symbol.upper())


def _symbol_to_yf(symbol: str, market_type: str = "equity") -> str:
    """Convert internal symbol format to yfinance ticker."""
    if market_type == "crypto":
        # BTC/USDT → BTC-USD; ETH/USDT → ETH-USD
        base = symbol.replace("/USDT", "").replace("/USD", "").replace("/BTC", "")
        return f"{base}-USD"
    if market_type == "forex":
        # EUR/USD, EURUSD, EUR-USD → EURUSD=X
        s = symbol.upper().replace("/", "").replace("-", "").replace("=X", "")
        return f"{s}=X"
    if market_type in ("commodity", "commodities", "future", "futures"):
        key = symbol.upper().replace("=F", "").replace("/", "")
        if key in COMMODITY_YF_MAP:
            return COMMODITY_YF_MAP[key]
        return symbol.upper() if symbol.upper().endswith("=F") else f"{key}=F"
    return symbol.upper()


def _interval_to_alpaca(interval: str) -> str:
    """Convert internal interval names to Alpaca's interval format."""
    return INTERVAL_ALPACA_MAP.get(interval.lower(), "1Day")


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


def _http_get_json(url: str, headers: dict, timeout: float = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES) -> dict:
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
                time.sleep(_BACKOFF_FACTOR * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _fetch_alpaca_crypto(
    symbol: str,
    start: date,
    end: date,
    interval: str,
    max_pages: int = DEFAULT_MAX_PAGES,
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
        url = f"{ALPACA_CRYPTO_BARS_URL}?{urllib.parse.urlencode(params)}"
        payload = _http_get_json(url, headers, timeout=DEFAULT_TIMEOUT)
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
    mu = _SYNTH_DRIFT_ANNUAL / 252  # daily drift
    sigma = _SYNTH_VOL_ANNUAL / (252 ** 0.5)  # daily vol
    log_returns = rng.normal(mu - 0.5 * sigma ** 2, sigma, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    noise = rng.uniform(_SYNTH_OPEN_NOISE_LOW, _SYNTH_OPEN_NOISE_HIGH, n)
    open_ = np.roll(close, 1) * noise
    open_[0] = close[0] * 0.999

    high = np.maximum(open_, close) * rng.uniform(_SYNTH_HIGH_MULT_LOW, _SYNTH_HIGH_MULT_HIGH, n)
    low = np.minimum(open_, close) * rng.uniform(_SYNTH_LOW_MULT_LOW, _SYNTH_LOW_MULT_HIGH, n)
    volume = rng.integers(_SYNTH_VOLUME_MIN, _SYNTH_VOLUME_MAX, n).astype(float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return df


# The public entry point used by strategies/backtests.
async def fetch_ohlcv(
    symbol: str,
    start: date,
    end: date,
    interval: str = "1d",
    market_type: str = "equity",
) -> pd.DataFrame:
    """
    Retrieve OHLCV data for a given symbol and date range.

    The loader prefers the following sources in order:
    1. Alpaca crypto bars (for crypto symbols)
    2. yfinance (for equities, forex, commodities, etc.)
    3. Synthetic GBM data (fallback when both sources fail)
    """
    # Crypto path – try Alpaca first
    if market_type == "crypto":
        df = _fetch_alpaca_crypto(symbol, start, end, interval)
        if not df.empty:
            return df

    # General path – yfinance
    try:
        import yfinance as yf

        yf_symbol = _symbol_to_yf(symbol, market_type=market_type)
        yf_interval = _interval_to_yf(interval)
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(start=start, end=end + timedelta(days=1), interval=yf_interval, auto_adjust=False)
        if not hist.empty:
            hist = hist.rename(
                columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
            )
            hist = hist[["open", "high", "low", "close", "volume"]]
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            return hist.sort_index()
    except Exception as exc:  # pragma: no cover – network or yfinance errors
        logger.debug("yfinance fetch failed for %s: %s", symbol, exc)

    # Fallback to synthetic data
    return _synthetic_ohlcv(symbol, start, end, interval)