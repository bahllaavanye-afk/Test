"""Market data endpoints: quotes, historical OHLCV, news, earnings."""
from fastapi import APIRouter, Depends, Query, HTTPException
from app.api.deps import get_current_user
from app.models.user import User
from app.config import settings
import yfinance as yf
import asyncio
import httpx

router = APIRouter(prefix="/market-data", tags=["market_data"])

ALPACA_DATA_URL = "https://data.alpaca.markets"


def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def _sentiment_score(text: str) -> float:
    """Simple keyword-based sentiment scoring clamped to [-1, 1]."""
    positive_words = ["beat", "surge", "rally", "record", "strong", "upgrade", "buy", "profit", "growth", "bullish"]
    negative_words = ["miss", "drop", "fall", "cut", "weak", "downgrade", "sell", "loss", "decline", "bearish"]
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
    """Get live quote for a single symbol. Tries Alpaca first, falls back to yfinance."""
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
                mid = round((float(bid) + float(ask)) / 2, 4) if bid and ask else None
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
    except Exception:
        pass

    # Fallback: yfinance
    loop = asyncio.get_running_loop()
    ticker = await loop.run_in_executor(None, lambda: yf.Ticker(sym_upper).fast_info)
    last = getattr(ticker, "last_price", None)
    return {
        "symbol": sym_upper,
        "bid_price": None,
        "ask_price": None,
        "bid_size": None,
        "ask_size": None,
        "mid_price": last,
        "last": last,
        "timestamp": None,
        "source": "yfinance",
        "market_cap": getattr(ticker, "market_cap", None),
        "currency": getattr(ticker, "currency", None),
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
                    mid = round((float(bid) + float(ask)) / 2, 4) if bid and ask else None
                    result.append({
                        "symbol": sym,
                        "bid_price": float(bid) if bid else None,
                        "ask_price": float(ask) if ask else None,
                        "bid_size": int(bid_size) if bid_size else None,
                        "ask_size": int(ask_size) if ask_size else None,
                        "mid_price": mid,
                        "last": mid,
                        "timestamp": ts,
                    })
                return result
    except Exception:
        pass

    return []


@router.get("/history/{symbol}")
async def get_history(
    symbol: str,
    interval: str = Query("1d", enum=["1m", "5m", "15m", "1h", "4h", "1d"]),
    period: str = Query("1y", enum=["1mo", "3mo", "6mo", "1y", "2y", "5y"]),
    current_user: User = Depends(get_current_user),
):
    loop = asyncio.get_running_loop()
    hist = await loop.run_in_executor(
        None, lambda: yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    )
    if hist.empty:
        return []
    hist = hist.reset_index()
    records = []
    for _, row in hist.iterrows():
        records.append({
            "time": str(row.get("Datetime", row.get("Date", ""))),
            "open": float(row.get("Open", 0)),
            "high": float(row.get("High", 0)),
            "low": float(row.get("Low", 0)),
            "close": float(row.get("Close", 0)),
            "volume": float(row.get("Volume", 0)),
        })
    return records


# ─── News ─────────────────────────────────────────────────────────────────────

@router.get("/news")
async def get_news(
    symbols: str = Query("SPY", description="Comma-separated symbols"),
    limit: int = Query(20, ge=1, le=50),
    current_user: User = Depends(get_current_user),
):
    """Proxy Alpaca News API and add keyword-based sentiment scoring."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{ALPACA_DATA_URL}/v1beta1/news",
                params={"symbols": symbols, "limit": limit},
                headers=_alpaca_headers(),
            )
            if resp.status_code != 200:
                return {"news": [], "error": f"Alpaca returned {resp.status_code}", "data_source": "unavailable"}

            raw = resp.json()
            articles = raw if isinstance(raw, list) else raw.get("news", [])

            result = []
            for item in articles:
                headline = item.get("headline", "")
                summary = item.get("summary", "")
                score = _sentiment_score(f"{headline} {summary}")
                result.append({
                    "id": item.get("id"),
                    "headline": headline,
                    "summary": summary,
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "author": item.get("author", ""),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "symbols": item.get("symbols", []),
                    "sentiment_score": round(score, 3),
                })
            return {"news": result, "data_source": "alpaca"}

    except httpx.TimeoutException:
        return {"news": [], "error": "Request timed out", "data_source": "unavailable"}
    except Exception as exc:
        return {"news": [], "error": str(exc), "data_source": "unavailable"}


# ─── Earnings Calendar ────────────────────────────────────────────────────────

@router.get("/earnings")
async def get_earnings(
    symbols: str = Query("AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,SPY"),
    current_user: User = Depends(get_current_user),
):
    """Proxy Alpaca corporate actions earnings data. Returns empty if not available (premium)."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{ALPACA_DATA_URL}/v1beta1/corporate-actions",
                params={"types": "earnings", "symbols": symbols},
                headers=_alpaca_headers(),
            )
            if resp.status_code == 403 or resp.status_code == 402:
                return {"earnings": [], "data_source": "unavailable", "reason": "premium_required"}
            if resp.status_code != 200:
                return {"earnings": [], "data_source": "unavailable", "reason": f"status_{resp.status_code}"}

            raw = resp.json()
            items = raw if isinstance(raw, list) else raw.get("earnings", raw.get("corporate_actions", []))

            result = []
            for item in items:
                eps_est = item.get("estimate_eps") or item.get("eps_estimate")
                eps_act = item.get("reported_eps") or item.get("eps_actual")
                surprise = None
                if eps_est is not None and eps_act is not None and eps_est != 0:
                    surprise = round((float(eps_act) - float(eps_est)) / abs(float(eps_est)) * 100, 2)
                result.append({
                    "symbol": item.get("symbol", ""),
                    "report_date": item.get("report_date") or item.get("date"),
                    "fiscal_year": item.get("fiscal_year") or item.get("fiscal_year_ending"),
                    "fiscal_quarter": item.get("fiscal_quarter") or item.get("quarter"),
                    "estimate_eps": float(eps_est) if eps_est is not None else None,
                    "reported_eps": float(eps_act) if eps_act is not None else None,
                    "surprise_pct": surprise,
                })
            return {"earnings": result, "data_source": "alpaca" if result else "unavailable"}

    except httpx.TimeoutException:
        return {"earnings": [], "data_source": "unavailable", "reason": "timeout"}
    except Exception as exc:
        return {"earnings": [], "data_source": "unavailable", "reason": str(exc)}
