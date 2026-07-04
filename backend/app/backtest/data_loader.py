"""OHLCV data loader with yfinance (free) as primary source.

Strategies and backtests call fetch_ohlcv() — it's entirely offline,
no broker keys required. yfinance pulls from Yahoo Finance for free.
"""
from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta

import pandas as pd

from app.utils.logging import logger

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Interval mappings
INTERVAL_TO_YF_MAP: dict[str, str] = {
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
DEFAULT_YF_INTERVAL = "1d"

INTERVAL_TO_ALPACA_MAP: dict[str, str] = {
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
DEFAULT_ALPACA_INTERVAL = "1Day"

# Commodity ticker mapping
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

# Alpaca crypto data source
ALPACA_CRYPTO_BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
ALPACA_DEFAULT_MAX_PAGES = 25
ALPACA_BAR_LIMIT = 10000
ALPACA_SORT_ORDER = "asc"
ALPACA_USER_AGENT = "QuantEdge/1.0"

# HTTP helper defaults
HTTP_TIMEOUT_DEFAULT = 20.0
HTTP_RETRIES_DEFAULT = 1

# Synthetic data generation parameters
ANNUAL_DRIFT = 0.10  # 10% annual drift
ANNUAL_VOL = 0.15    # 15% annual volatility
TRADING_DAYS = 252
VOLUME_MIN = 1_000_000
VOLUME_MAX = 50_000_000
NOISE_LOWER = 0.998
NOISE_UPPER = 1.002
HIGH_MULT_LOWER = 1.000
HIGH_MULT_UPPER = 1.010
LOW_MULT_LOWER = 0.990
LOW_MULT_UPPER = 1.000

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _interval_to_yf(interval: str) -> str:
    """Convert internal interval names to yfinance format."""
    return INTERVAL_TO_YF_MAP.get(interval.lower(), DEFAULT_YF_INTERVAL)


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
    """Map internal interval to Alpaca interval string."""
    return INTERVAL_TO_ALPACA_MAP.get(interval.lower(), DEFAULT_ALPACA_INTERVAL)


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


def _http_get_json(
    url: str,
    headers: dict,
    timeout: float = HTTP_TIMEOUT_DEFAULT,
    retries: int = HTTP_RETRIES_DEFAULT,
) -> dict:
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
    symbol: str,
    start: date,
    end: date,
    interval: str,
    max_pages: int = ALPACA_DEFAULT_MAX_PAGES,
) -> pd.DataFrame:
    """Fetch crypto OHLCV from Alpaca's public crypto bars API.

    Returns a tz-naive DataFrame [open, high, low, close, volume] sorted ascending,
    or an empty DataFrame if no bars are returned. Follows next_page_token.
    """
    import urllib.parse

    pair = _symbol_to_alpaca_crypto(symbol)
    timeframe = _interval_to_alpaca(interval)
    headers = {"User-Agent": ALPACA_USER_AGENT, "Accept": "application/json"}
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
            "limit": ALPACA_BAR_LIMIT,
            "sort": ALPACA_SORT_ORDER,
        }
        if page_token:
            params["page_token"] = page_token
        url = f"{ALPACA_CRYPTO_BARS_URL}?{urllib.parse.urlencode(params)}"
        payload = _http_get_json(url, headers, timeout=HTTP_TIMEOUT_DEFAULT)
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
    mu = ANNUAL_DRIFT / TRADING_DAYS
    sigma = ANNUAL_VOL / (TRADING_DAYS ** 0.5)
    log_returns = rng.normal(mu - 0.5 * sigma ** 2, sigma, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    noise = rng.uniform(NOISE_LOWER, NOISE_UPPER, n)
    open_ = np.roll(close, 1) * noise
    open_[0] = close[0] * 0.999
    high = np.maximum(open_, close) * rng.uniform(HIGH_MULT_LOWER, HIGH_MULT_UPPER, n)
    low = np.minimum(open_, close) * rng.uniform(LOW_MULT_LOWER, LOW_MULT_UPPER, n)
    volume = rng.integers(VOLUME_MIN, VOLUME_MAX, n).astype(float)

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