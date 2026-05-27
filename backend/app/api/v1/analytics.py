"""Analytics and performance metrics endpoints."""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from datetime import datetime, timezone, timedelta
from typing import Optional
import httpx
import pandas as pd

from app.database import get_db
from app.api.deps import get_current_user
from app.models.trade import Trade
from app.models.slippage import SlippageRecord
from app.models.user import User
from app.models.account import Account
from app.models.position import Position
from app.models.order import Order
from app.config import settings

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/performance")
async def get_performance(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Aggregate trade performance stats."""
    result = await db.execute(
        select(
            func.count(Trade.id).label("total_trades"),
            func.avg(Trade.realized_pnl).label("avg_pnl"),
            func.sum(Trade.realized_pnl).label("total_pnl"),
        )
    )
    row = result.one()
    return {
        "total_trades": row.total_trades or 0,
        "avg_pnl": float(row.avg_pnl or 0),
        "total_pnl": float(row.total_pnl or 0),
    }


@router.get("/slippage")
async def get_slippage_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Average slippage by execution algorithm."""
    result = await db.execute(
        select(
            SlippageRecord.execution_algo,
            func.avg(SlippageRecord.slippage_bps).label("avg_bps"),
            func.count(SlippageRecord.id).label("count"),
        ).group_by(SlippageRecord.execution_algo)
    )
    rows = result.all()
    return [{"algo": r.execution_algo, "avg_bps": round(float(r.avg_bps or 0), 2), "count": r.count} for r in rows]


@router.get("/attribution")
async def get_pnl_attribution(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """P&L broken down by strategy — the #1 feature missing from open-source bots."""
    result = await db.execute(
        select(
            Trade.strategy_name,
            func.count(Trade.id).label("trades"),
            func.sum(Trade.realized_pnl).label("total_pnl"),
            func.avg(Trade.realized_pnl).label("avg_pnl"),
            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("wins"),
        ).group_by(Trade.strategy_name).order_by(func.sum(Trade.realized_pnl).desc())
    )
    rows = result.all()
    out = []
    for r in rows:
        total = float(r.total_pnl or 0)
        trades = r.trades or 0
        wins = r.wins or 0
        out.append({
            "strategy": r.strategy_name or "manual",
            "trades": trades,
            "total_pnl": round(total, 2),
            "avg_pnl": round(float(r.avg_pnl or 0), 2),
            "win_rate": round(wins / max(trades, 1), 3),
        })
    return out


@router.get("/macro")
async def get_macro_signals(current_user: User = Depends(get_current_user)):
    """Current macro environment signals from FRED (free, no API key)."""
    from app.ml.features.macro_signals import get_macro_snapshot_cached
    return await get_macro_snapshot_cached()


@router.get("/sentiment")
async def get_reddit_sentiment_endpoint(
    tickers: str | None = None,
    current_user: User = Depends(get_current_user),
):
    """Reddit WallStreetBets sentiment from Apewisdom (free, no key required)."""
    from app.ml.features.macro_signals import get_reddit_sentiment
    ticker_list = tickers.split(",") if tickers else None
    return await get_reddit_sentiment(ticker_list)


# ─── Correlation Matrix ───────────────────────────────────────────────────────

DEFAULT_SYMBOLS = ["SPY", "QQQ", "GLD", "TLT", "AAPL", "BTC/USD"]

ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"


async def _fetch_alpaca_bars(symbols: list[str], days: int) -> dict[str, list[float]]:
    """Fetch daily close prices from Alpaca for the given symbols.

    Returns a dict mapping symbol -> list of close prices (oldest first).
    Symbols that fail to fetch are omitted from the result.
    """
    start_dt = (datetime.now(timezone.utc) - timedelta(days=days + 10)).strftime("%Y-%m-%d")
    # Alpaca doesn't carry BTC/USD in the stock bars endpoint — filter to equity-like symbols
    equity_symbols = [s for s in symbols if "/" not in s]
    if not equity_symbols:
        return {}

    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    params = {
        "symbols": ",".join(equity_symbols),
        "timeframe": "1Day",
        "start": start_dt,
        "limit": days + 10,
        "feed": "iex",
        "adjustment": "raw",
    }

    prices: dict[str, list[float]] = {}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(ALPACA_DATA_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            bars_map = data.get("bars", {})
            for sym, bars in bars_map.items():
                prices[sym] = [float(b["c"]) for b in bars]
    except Exception:
        pass
    return prices


@router.get("/correlation")
async def get_correlation_matrix(
    account_id: Optional[str] = Query(None),
    days: int = Query(30, ge=5, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Compute pairwise Pearson correlation matrix of daily returns
    for the user's current open positions.
    """
    # Gather symbols from user's positions
    symbols: list[str] = []
    try:
        acct_q = select(Account).where(Account.user_id == current_user.id, Account.is_active == True)
        if account_id:
            acct_q = acct_q.where(Account.id == account_id)
        acct_result = await db.execute(acct_q)
        accounts = acct_result.scalars().all()

        if accounts:
            account_ids = [a.id for a in accounts]
            pos_result = await db.execute(
                select(Position.symbol).where(Position.account_id.in_(account_ids)).distinct()
            )
            symbols = [row[0] for row in pos_result.all()]
    except Exception:
        symbols = []

    if not symbols:
        symbols = DEFAULT_SYMBOLS

    # Fetch price data from Alpaca (equity symbols only)
    prices_map = await _fetch_alpaca_bars(symbols, days)

    if not prices_map:
        return {
            "symbols": symbols,
            "matrix": [],
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "error": "Unable to fetch price data from Alpaca. Check API credentials.",
        }

    # Keep only symbols we have data for
    available_symbols = sorted(prices_map.keys())

    # Build DataFrame of close prices
    series_dict: dict[str, pd.Series] = {}
    for sym in available_symbols:
        closes = prices_map[sym]
        if len(closes) >= 5:
            series_dict[sym] = pd.Series(closes, dtype=float)

    if len(series_dict) < 2:
        return {
            "symbols": available_symbols,
            "matrix": [[1.0]] if len(series_dict) == 1 else [],
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    # Align series by index (use shortest length)
    min_len = min(len(s) for s in series_dict.values())
    df = pd.DataFrame({sym: s.iloc[-min_len:].values for sym, s in series_dict.items()})

    # Daily returns
    returns = df.pct_change().dropna()

    # Pearson correlation
    corr = returns.corr(method="pearson")
    final_symbols = list(corr.columns)
    matrix = [[round(float(corr.loc[r, c]), 4) for c in final_symbols] for r in final_symbols]

    return {
        "symbols": final_symbols,
        "matrix": matrix,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Tax Lots ─────────────────────────────────────────────────────────────────

ALPACA_QUOTES_URL = "https://data.alpaca.markets/v2/stocks/quotes/latest"


async def _fetch_latest_price(symbol: str) -> Optional[float]:
    """Try to get the latest ask price for a symbol from Alpaca."""
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                ALPACA_QUOTES_URL,
                headers=headers,
                params={"symbols": symbol, "feed": "iex"},
            )
            resp.raise_for_status()
            data = resp.json()
            quote = data.get("quotes", {}).get(symbol)
            if quote:
                # midpoint of bid/ask
                bid = float(quote.get("bp", 0))
                ask = float(quote.get("ap", 0))
                if ask > 0:
                    return (bid + ask) / 2.0
    except Exception:
        pass
    return None


@router.get("/tax-lots/{symbol}")
async def get_tax_lots(
    symbol: str,
    account_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Compute open tax lots for a symbol using FIFO matching of buys vs sells.
    Returns unrealized P&L, holding period, and HIFO/FIFO/LIFO recommendation.
    """
    # Resolve account IDs for this user
    acct_q = select(Account.id).where(Account.user_id == current_user.id, Account.is_active == True)
    if account_id:
        acct_q = acct_q.where(Account.id == account_id)
    acct_result = await db.execute(acct_q)
    account_ids = [row[0] for row in acct_result.all()]

    if not account_ids:
        raise HTTPException(status_code=404, detail="No accounts found for this user.")

    # Fetch all filled buy orders for this symbol
    buy_q = (
        select(Order)
        .where(
            Order.account_id.in_(account_ids),
            Order.symbol == symbol.upper(),
            Order.side == "buy",
            Order.status == "filled",
            Order.filled_qty > 0,
        )
        .order_by(Order.filled_at.asc())
    )
    buy_result = await db.execute(buy_q)
    buys = buy_result.scalars().all()

    # Fetch all filled sell orders
    sell_q = (
        select(Order)
        .where(
            Order.account_id.in_(account_ids),
            Order.symbol == symbol.upper(),
            Order.side == "sell",
            Order.status == "filled",
            Order.filled_qty > 0,
        )
        .order_by(Order.filled_at.asc())
    )
    sell_result = await db.execute(sell_q)
    sells = sell_result.scalars().all()

    if not buys:
        return {
            "symbol": symbol.upper(),
            "lots": [],
            "total_unrealized_pnl": 0.0,
            "recommended_method": "FIFO",
            "tax_savings_hifo_vs_fifo": 0.0,
        }

    # FIFO matching: consume sell quantity against earliest buys first
    # Build mutable lot list: each entry is [qty_remaining, cost_basis_per_share, filled_at, order_id]
    lots_raw = [
        {
            "qty": float(o.filled_qty),
            "cost": float(o.avg_fill_price) if o.avg_fill_price else 0.0,
            "acquired_at": o.filled_at,
            "order_id": o.id,
        }
        for o in buys
    ]

    # Total sell quantity to consume
    total_sold = sum(float(o.filled_qty) for o in sells)

    remaining_sell = total_sold
    for lot in lots_raw:
        if remaining_sell <= 0:
            break
        consumed = min(lot["qty"], remaining_sell)
        lot["qty"] -= consumed
        remaining_sell -= consumed

    # Open lots: those with remaining quantity > 0
    open_lots = [l for l in lots_raw if l["qty"] > 1e-9]

    if not open_lots:
        return {
            "symbol": symbol.upper(),
            "lots": [],
            "total_unrealized_pnl": 0.0,
            "recommended_method": "FIFO",
            "tax_savings_hifo_vs_fifo": 0.0,
        }

    # Fetch current price — fall back to latest fill price in buy orders
    current_price = await _fetch_latest_price(symbol.upper())
    if current_price is None and buys:
        last_fill = buys[-1].avg_fill_price
        current_price = float(last_fill) if last_fill else None

    now = datetime.now(timezone.utc)
    result_lots = []
    for i, lot in enumerate(open_lots):
        qty = float(lot["qty"])
        cost = float(lot["cost"])
        acquired = lot["acquired_at"]
        # Ensure timezone-aware
        if acquired and acquired.tzinfo is None:
            acquired = acquired.replace(tzinfo=timezone.utc)

        holding_days = (now - acquired).days if acquired else 0
        is_long_term = holding_days > 365

        if current_price is not None and cost > 0:
            unrealized_pnl = (current_price - cost) * qty
            unrealized_pct = ((current_price - cost) / cost) * 100.0
        else:
            unrealized_pnl = None
            unrealized_pct = None

        result_lots.append({
            "lot_id": lot["order_id"],
            "symbol": symbol.upper(),
            "quantity": round(qty, 8),
            "cost_basis": round(cost, 4),
            "acquired_date": acquired.isoformat() if acquired else None,
            "current_price": round(current_price, 4) if current_price is not None else None,
            "unrealized_pnl": round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
            "unrealized_pct": round(unrealized_pct, 2) if unrealized_pct is not None else None,
            "holding_days": holding_days,
            "is_long_term": is_long_term,
        })

    total_unrealized_pnl = sum(
        l["unrealized_pnl"] for l in result_lots if l["unrealized_pnl"] is not None
    )

    # Recommend method based on gain/loss situation
    # HIFO (highest cost first) minimizes gains when selling — best when lots have gains
    # LIFO may be better in declining markets
    # Simple heuristic: if total P&L > 0 → HIFO saves the most taxes
    #                   if total P&L < 0 → FIFO (harvest losses early)
    if total_unrealized_pnl > 0:
        recommended_method = "HIFO"
    elif total_unrealized_pnl < 0:
        recommended_method = "FIFO"
    else:
        recommended_method = "FIFO"

    # Compute HIFO vs FIFO tax savings estimate (if we were to sell all open lots)
    # FIFO: sell lowest-cost lots first → highest gains
    # HIFO: sell highest-cost lots first → lowest gains
    if current_price is not None:
        fifo_lots_by_cost_asc = sorted(result_lots, key=lambda x: x["cost_basis"])
        hifo_lots_by_cost_desc = sorted(result_lots, key=lambda x: x["cost_basis"], reverse=True)

        fifo_gain = sum((current_price - l["cost_basis"]) * l["quantity"] for l in fifo_lots_by_cost_asc)
        hifo_gain = sum((current_price - l["cost_basis"]) * l["quantity"] for l in hifo_lots_by_cost_desc)
        # Tax savings = difference in taxable gain (assume ~20% cap gains rate for illustration)
        tax_savings_hifo_vs_fifo = round((fifo_gain - hifo_gain) * 0.20, 2)
    else:
        tax_savings_hifo_vs_fifo = 0.0

    return {
        "symbol": symbol.upper(),
        "lots": result_lots,
        "total_unrealized_pnl": round(float(total_unrealized_pnl), 2),
        "recommended_method": recommended_method,
        "tax_savings_hifo_vs_fifo": tax_savings_hifo_vs_fifo,
    }
