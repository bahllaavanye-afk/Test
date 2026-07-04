"""OHLCV data loader with yfinance (free) as primary source.

Strategies and backtests call fetch_ohlcv() — it's entirely offline,
no broker keys required. yfinance pulls from Yahoo Finance for free.
"""
from __future__ import annotations

import asyncio
import os
import urllib.parse
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from app.utils.logging import logger


def _interval_to_yf(interval: str) -> str:
    """Convert internal interval names to yfinance format."""
    _MAP = {
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
    return _MAP.get(interval.lower(), "1d")


# Friendly commodity names → yfinance continuous-future tickers.
_COMMODITY_YF = {
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
        if key in _COMMODITY_YF:
            return _COMMODITY_YF[key]
        return symbol.upper() if symbol.upper().endswith("=F") else f"{key}=F"
    return symbol.upper()


# ── Alpaca crypto market data (free, keyless public endpoint) ──────────────────
_ALPACA_CRYPTO_BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"

_INTERVAL_TO_ALPACA = {
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


def _interval_to_alpaca(interval: str) -> str:
    """Map internal interval string to Alpaca's interval identifier."""
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


def _http_get_json(
    url: str, headers: Dict[str, str], timeout: float = 20.0, retries: int = 1
) -> Dict[str, Any]:
    """Minimal stdlib JSON GET with a light retry (kept tiny + patchable for tests)."""
    import json
    import time
    import urllib.request

    last_exc: Optional[Exception] = None
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


def _build_alpaca_params(
    pair: str,
    timeframe: str,
    start: date,
    end: date,
    page_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct query parameters for Alpaca crypto bars request."""
    params = {
        "symbols": pair,
        "timeframe": timeframe,
        "start": start.isoformat(),
        # Alpaca's `end` is INCLUSIVE of bar timestamps. Use end-of-day so we
        # capture every bar on `end` (the 00:00 daily bar *and* all intraday
        # bars) without pulling the next day's bar.
        "end": f"{end.isoformat()}T23:59:59Z",
        "limit": 10000,
        "sort": "asc",
    }
    if page_token:
        params["page_token"] = page_token
    return params


def _parse_alpaca_payload(payload: Dict[str, Any], pair: str) -> List[Dict[str, Any]]:
    """Extract bar rows for the requested pair from Alpaca's response payload."""
    return (payload.get("bars") or {}).get(pair, [])


def _fetch_alpaca_crypto(
    symbol: str, start: date, end: date, interval: str, max_pages: int = 25
) -> pd.DataFrame:
    """Fetch crypto OHLCV from Alpaca's public crypto bars API.

    Returns a tz‑naive DataFrame [open, high, low, close, volume] sorted ascending,
    or an empty DataFrame if no bars are returned. Follows next_page_token.
    """
    pair = _symbol_to_alpaca_crypto(symbol)
    timeframe = _interval_to_alpaca(interval)

    headers = {"User-Agent": "QuantEdge/1.0", "Accept": "application/json"}
    # Crypto bars are public, but sending keys (when present) raises rate limits.
    key = os.environ.get("ALPACA_API_KEY", "")
    sec = os.environ.get("ALPACA_SECRET_KEY", "")
    if key and sec and key != "test-key":
        headers["APCA-API-KEY-ID"] = key
        headers["APCA-API-SECRET-KEY"] = sec

    rows: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    for _ in range(max_pages):
        params = _build_alpaca_params(pair, timeframe, start, end, page_token)
        url = f"{_ALPACA_CRYPTO_BARS_URL}?{urllib.parse.urlencode(params)}"
        payload = _http_get_json(url, headers, timeout=20.0)
        rows.extend(_parse_alpaca_payload(payload, pair))
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    else:
        # max_pages exhausted without consuming the last page → bars were truncated
        logger.warning(
            f"Alpaca crypto: hit max_pages={max_pages} for {pair} ({interval}); "
            "older bars may be truncated — widen max_pages or narrow the range"
        )

    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "t": "ts",
            }
        )
        .assign(ts=lambda d: pd.to_datetime(d["ts"], utc=True).dt.tz_localize(None))
        .set_index("ts")[["open", "high", "low", "close", "volume"]]
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
    Returns realistic‑looking daily bars with drift ≈ 10 % pa, vol ≈ 15 % pa.
    """
    import numpy as np

    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    if n < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(sum(ord(c) for c in symbol))
    mu = 0.10 / 252  # 10 % annual drift
    sigma = 0.15 / (252 ** 0.5)  # 15 % annual vol

    log_returns = rng.normal(mu - 0.5 * sigma ** 2, sigma, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    noise = rng.uniform(0.998, 1.002, n)
    open_ = np.roll(close, 1) * noise
    open_[0] = close[0] * 0.999

    high = np.maximum(open_, close) * rng.uniform(1.000, 1.010, n)
    low = np.minimum(open_, close) * rng.uniform(0.990, 1.000, n)
    volume = rng.integers(1_000_000, 50_000_000, n).astype(float)

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
    df.index.name = None
    return df.astype(float)


def fetch_ohlcv(
    symbol: str,
    start: date,
    end: date,
    interval: str = "1d",
    market_type: str = "equity",
) -> pd.DataFrame:
    """Public entry point for OHLCV retrieval.

    The function attempts, in order:
    1. Alpaca crypto bars (if ``market_type`` is ``'crypto'``).
    2. yfinance for equities, forex, commodities, etc.
    3. Synthetic data as a last‑resort fallback.

    Returns a DataFrame with columns ``['open','high','low','close','volume']``
    indexed by naive ``datetime`` objects in ascending order.
    """
    # 1️⃣ Crypto via Alpaca – the only source that provides real‑time crypto bars
    # without requiring API credentials.
    if market_type == "crypto":
        try:
            df = _fetch_alpaca_crypto(symbol, start, end, interval)
            if not df.empty:
                return df
            logger.debug("Alpaca returned empty DataFrame for %s; falling back.", symbol)
        except Exception as exc:  # pragma: no cover – exercised via integration tests
            logger.warning("Alpaca fetch failed for %s: %s", symbol, exc)

    # 2️⃣ yfinance – works for equities, forex, commodities, and also crypto (synthetic).
    try:
        import yfinance as yf

        yf_symbol = _symbol_to_yf(symbol, market_type)
        yf_interval = _interval_to_yf(interval)

        # yfinance's `end` is exclusive, so we add one day to include the final day.
        yf_end = end + timedelta(days=1)

        data = yf.download(
            yf_symbol,
            start=start,
            end=yf_end,
            interval=yf_interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if data.empty:
            raise ValueError("Empty data from yfinance")
        df = (
            data[["Open", "High", "Low", "Close", "Volume"]]
            .rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            .astype(float)
        )
        df.index.name = None
        return df
    except Exception as exc:  # pragma: no cover – exercised via integration tests
        logger.warning("yfinance fetch failed for %s (%s): %s", symbol, market_type, exc)

    # 3️⃣ Synthetic fallback – guarantees that a DataFrame is always returned.
    logger.info("Falling back to synthetic OHLCV for %s", symbol)
    return _synthetic_ohlcv(symbol, start, end, interval)