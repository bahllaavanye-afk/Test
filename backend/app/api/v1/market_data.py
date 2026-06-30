"""Market data endpoints: quotes, historical OHLCV, news, earnings, IV Rank, PCR."""
from datetime import datetime, timezone, timedelta

import httpx
from fastapi import APIRouter, Depends, Query, HTTPException

from app.api.deps import get_current_user
from app.config import settings
from app.models.user import User
from app.utils.logging import logger

router = APIRouter(prefix="/market-data", tags=["market_data"])

ALPACA_DATA_URL = "https://data.alpaca.markets"

# Crypto base symbols — any symbol containing these is routed to the crypto endpoint
_CRYPTO_BASES = {
    "BTC",
    "ETH",
    "SOL",
    "ADA",
    "AVAX",
    "DOT",
    "MATIC",
    "LINK",
    "UNI",
    "LTC",
    "XRP",
    "DOGE",
    "SHIB",
    "BCH",
}


def _alpaca_headers() -> dict:
    """Build Alpaca authentication headers."""
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def _is_crypto(symbol: str) -> bool:
    """Return True if symbol looks like a crypto asset."""
    sym = symbol.upper()
    if "/" in sym:
        return True
    for base in _CRYPTO_BASES:
        if sym.startswith(base) or sym.endswith(base):
            return True
    return False


def _interval_to_alpaca(interval: str) -> str:
    """Map internal interval representation to Alpaca's format."""
    mapping = {
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "1h": "1Hour",
        "4h": "4Hour",
        "1d": "1Day",
        "1w": "1Week",
    }
    return mapping.get(interval, "1Day")


def _period_to_start(period: str) -> str:
    """Convert a period string (e.g., '1mo') to an ISO‑8601 start timestamp."""
    days_map = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
    }
    days = days_map.get(period, 365)
    start_dt = datetime.now(timezone.utc) - timedelta(days=days)
    return start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


async def _fetch_alpaca_bars(
    symbol: str, timeframe: str, start: str, limit: int = 1000
) -> list[dict]:
    """
    Fetch OHLCV bars from Alpaca for stocks or crypto.
    Returns list of dicts with keys: time, open, high, low, close, volume, vwap
    """
    sym_upper = symbol.upper()
    headers = _alpaca_headers()

    async with httpx.AsyncClient(timeout=15.0) as client:
        if _is_crypto(sym_upper):
            url = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
            params = {
                "symbols": sym_upper,
                "timeframe": timeframe,
                "start": start,
                "limit": limit,
            }
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Alpaca crypto bars error: {resp.status_code} {resp.text[:200]}",
                )
            data = resp.json()
            bars_by_sym = data.get("bars", {})
            raw_bars = bars_by_sym.get(sym_upper, [])
        else:
            url = f"{ALPACA_DATA_URL}/v2/stocks/{sym_upper}/bars"
            params = {"timeframe": timeframe, "start": start, "limit": limit}
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Alpaca bars error: {resp.status_code} {resp.text[:200]}",
                )
            data = resp.json()
            raw_bars = data.get("bars", [])

    records = []
    for bar in raw_bars:
        records.append(
            {
                "time": bar.get("t", ""),
                "open": float(bar.get("o", 0)),
                "high": float(bar.get("h", 0)),
                "low": float(bar.get("l", 0)),
                "close": float(bar.get("c", 0)),
                "volume": float(bar.get("v", 0)),
                "vwap": float(bar.get("vw", 0)),
            }
        )
    return records


def _sentiment_score(text: str) -> float:
    """Simple keyword‑based sentiment scoring clamped to [-1, 1]."""
    positive_words = [
        "beat",
        "surge",
        "rally",
        "record",
        "strong",
        "upgrade",
        "buy",
        "profit",
        "growth",
        "bullish",
    ]
    negative_words = [
        "miss",
        "drop",
        "fall",
        "cut",
        "weak",
        "downgrade",
        "sell",
        "loss",
        "decline",
        "bearish",
    ]
    lower = text.lower()
    pos = sum(lower.count(w) for w in positive_words)
    neg = sum(lower.count(w) for w in negative_words)
    denom = max(pos + neg, 1)
    raw = (pos - neg) / denom
    return max(-1.0, min(1.0, raw))


# ─── Existing endpoints ───────────────────────────────────────────────────────


@router.get("/quote/{symbol}")
async def get_quote(
    symbol: str,
    current_user: User = Depends(get_current_user),
):
    """Get live quote for a single symbol via Alpaca."""
    sym_upper = symbol.upper()
    # Try Alpaca stocks quote
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{ALPACA_DATA_URL}/v2/stocks/{sym_upper}/quotes/latest",
                headers=_alpaca_headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                q = data.get("quote", {})
                bid = q.get("bp") or q.get("bid_price")
                ask = q.get("ap") or q.get("ask_price")
                bid_size = q.get("bs") or q.get("bid_size")
                ask_size = q.get("as") or q.get("ask_size")
                ts = q.get("t") or q.get("timestamp")
                mid = (
                    round((float(bid) + float(ask)) / 2, 4) if bid and ask else None
                )
                return {
                    "symbol": sym_upper,
                    "bid_price": float(bid) if bid else None,
                    "ask_price": float(ask) if ask else None,
                    "bid_size": int(bid_size) if bid_size else None,
                    "ask_size": int(ask_size) if ask_size else None,
                    "mid_price": mid,
                    "last": mid,
                    "timestamp": ts,
                    "source": "alpaca",
                }
    except Exception as exc:
        logger.debug("quote fetch (stocks) failed", symbol=sym_upper, error=str(exc))

    # Fallback: try latest bar price from Alpaca
    try:
        bars = await _fetch_alpaca_bars(
            sym_upper, "1Day", _period_to_start("1mo"), limit=1
        )
        if bars:
            last = bars[-1]["close"]
            return {
                "symbol": sym_upper,
                "bid_price": None,
                "ask_price": None,
                "bid_size": None,
                "ask_size": None,
                "mid_price": last,
                "last": last,
                "timestamp": bars[-1]["time"],
                "source": "alpaca_bars",
            }
    except Exception as exc:
        logger.debug(
            "quote fetch (bars fallback) failed", symbol=sym_upper, error=str(exc)
        )

    return {
        "symbol": sym_upper,
        "bid_price": None,
        "ask_price": None,
        "bid_size": None,
        "ask_size": None,
        "mid_price": None,
        "last": None,
        "timestamp": None,
        "source": "unavailable",
    }


@router.get("/quotes")
async def get_quotes_batch(
    symbols: str = Query(..., description="Comma-separated symbols, e.g. AAPL,MSFT,NVDA"),
    current_user: User = Depends(get_current_user),
):
    """Get live quotes for multiple symbols at once."""
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        return []

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{ALPACA_DATA_URL}/v2/stocks/quotes/latest",
                params={"symbols": ",".join(sym_list)},
                headers=_alpaca_headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                quotes = data.get("quotes", {})
                result = []
                for sym, q in quotes.items():
                    bid = q.get("bp") or q.get("bid_price")
                    ask = q.get("ap") or q.get("ask_price")
                    bid_size = q.get("bs") or q.get("bid_size")
                    ask_size = q.get("as") or q.get("ask_size")
                    ts = q.get("t") or q.get("timestamp")
                    mid = (
                        round((float(bid) + float(ask)) / 2, 4) if bid and ask else None
                    )
                    result.append(
                        {
                            "symbol": sym.upper(),
                            "bid_price": float(bid) if bid else None,
                            "ask_price": float(ask) if ask else None,
                            "bid_size": int(bid_size) if bid_size else None,
                            "ask_size": int(ask_size) if ask_size else None,
                            "mid_price": mid,
                            "last": mid,
                            "timestamp": ts,
                            "source": "alpaca",
                        }
                    )
                return result
    except Exception as exc:
        logger.debug(
            "quotes batch fetch failed",
            symbols=",".join(sym_list),
            error=str(exc),
        )

    # If the batch request fails, return an empty list to keep the response shape consistent.
    return []