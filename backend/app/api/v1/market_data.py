"""Market data endpoints: quotes, historical OHLCV."""
from fastapi import APIRouter, Depends, Query
from app.api.deps import get_current_user
from app.models.user import User
import yfinance as yf
import asyncio

router = APIRouter(prefix="/market-data", tags=["market_data"])


@router.get("/quote/{symbol}")
async def get_quote(
    symbol: str,
    current_user: User = Depends(get_current_user),
):
    loop = asyncio.get_event_loop()
    ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol).fast_info)
    return {
        "symbol": symbol,
        "last": getattr(ticker, "last_price", None),
        "market_cap": getattr(ticker, "market_cap", None),
        "currency": getattr(ticker, "currency", None),
    }


@router.get("/history/{symbol}")
async def get_history(
    symbol: str,
    interval: str = Query("1d", enum=["1m", "5m", "15m", "1h", "4h", "1d"]),
    period: str = Query("1y", enum=["1mo", "3mo", "6mo", "1y", "2y", "5y"]),
    current_user: User = Depends(get_current_user),
):
    loop = asyncio.get_event_loop()
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
