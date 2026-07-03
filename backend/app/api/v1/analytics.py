"""Analytics and performance metrics endpoints."""
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple
import asyncio

import numpy as np
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, desc

from app.database import get_db
from app.api.deps import get_current_user
from app.models.trade import Trade
from app.models.slippage import SlippageRecord
from app.models.user import User
from app.models.account import Account
from app.models.position import Position
from app.models.order import Order
from app.config import settings
from app.utils.logging import logger

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Simple in‑memory cache for user‑account mapping (TTL = 5 minutes)
_account_ids_cache: Dict[str, Tuple[float, List[str]]] = {}
_CACHE_TTL = 300.0  # seconds


async def _user_account_ids(db: AsyncSession, user_id: str) -> List[str]:
    """Return all account IDs owned by the given user with lightweight caching."""
    now_ts = asyncio.get_event_loop().time()
    cached = _account_ids_cache.get(user_id)
    if cached:
        ts, ids = cached
        if now_ts - ts < _CACHE_TTL:
            return ids

    result = await db.execute(select(Account.id).where(Account.user_id == user_id))
    ids = [row[0] for row in result.all()]
    _account_ids_cache[user_id] = (now_ts, ids)
    return ids


@router.get("/")
async def analytics_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """High-level analytics summary: available modules and quick stats."""
    try:
        trade_count_result = await db.execute(select(func.count()).select_from(Trade))
        trade_count = trade_count_result.scalar() or 0
    except Exception:
        trade_count = 0
    return {
        "modules": [
            "arb-opportunities",
            "performance",
            "slippage",
            "attribution",
            "macro",
            "sentiment",
            "correlation",
            "tearsheet",
            "equity-curve",
            "monthly-returns",
            "portfolio-greeks",
        ],
        "trade_count": trade_count,
        "tearsheet_available": trade_count > 0,
    }


@router.get("/arb-opportunities")
async def get_arb_opportunities(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return current arbitrage opportunities.

    Reads from the OHLCV table looking for cross‑exchange price discrepancies.
    Returns an empty list when no data is available rather than 404.
    """
    try:
        from app.models.market_data import OHLCV

        # Most recent snapshots (limit 200) ordered by timestamp
        result = await db.execute(
            select(OHLCV.symbol, OHLCV.exchange, OHLCV.close, OHLCV.ts)
            .order_by(desc(OHLCV.ts))
            .limit(200)
        )
        rows = result.all()

        # Group by symbol using a dict of lists
        by_symbol: Dict[str, List[Dict]] = {}
        for row in rows:
            by_symbol.setdefault(row.symbol, []).append(
                {
                    "exchange": row.exchange,
                    "price": float(row.close),
                    "ts": row.ts.isoformat() if row.ts else None,
                }
            )

        opportunities = []
        for symbol, entries in by_symbol.items():
            if len(entries) < 2:
                continue
            prices = [e["price"] for e in entries if e["price"] > 0]
            if len(prices) < 2:
                continue
            spread = max(prices) - min(prices)
            min_price = min(prices)
            spread_pct = (spread / min_price * 100) if min_price > 0 else 0.0
            if spread_pct > 0.05:  # surface only if >5bps spread
                opportunities.append(
                    {
                        "symbol": symbol,
                        "spread": round(spread, 6),
                        "spread_pct": round(spread_pct, 4),
                        "exchanges": entries,
                    }
                )
        return opportunities
    except Exception as exc:
        logger.warning("arb-opportunities endpoint failed", error=str(exc))
        return []


@router.get("/performance")
async def get_performance(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Aggregate trade performance stats — scoped to current user's accounts."""
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return {
            "total_trades": 0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": None,
            "max_drawdown": None,
        }

    result = await db.execute(
        select(
            func.count(Trade.id).label("total_trades"),
            func.avg(Trade.realized_pnl).label("avg_pnl"),
            func.sum(Trade.realized_pnl).label("total_pnl"),
            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("wins"),
        ).where(Trade.account_id.in_(account_ids))
    )
    row = result.one()
    total_trades = row.total_trades or 0
    wins = int(row.wins or 0)
    win_rate = round(wins / max(total_trades, 1), 4)

    # Compute Sharpe and max drawdown from daily PnL series
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    try:
        daily_result = await db.execute(
            select(
                func.date_trunc("day", Trade.closed_at).label("day"),
                func.sum(Trade.realized_pnl).label("daily_pnl"),
            )
            .where(
                Trade.account_id.in_(account_ids),
                Trade.closed_at >= datetime.now(timezone.utc) - timedelta(days=365),
                Trade.realized_pnl.isnot(None),
            )
            .group_by(func.date_trunc("day", Trade.closed_at))
            .order_by(func.date_trunc("day", Trade.closed_at))
        )
        daily_rows = daily_result.all()
        if len(daily_rows) >= 5:
            daily_pnls = np.array([float(r.daily_pnl) for r in daily_rows], dtype=np.float64)
            mean_r = daily_pnls.mean()
            std_r = daily_pnls.std(ddof=1)  # sample std
            if std_r > 0:
                sharpe_ratio = round(float(mean_r / std_r * (252 ** 0.5)), 4)

            cum = np.cumsum(daily_pnls)
            rolling_max = np.maximum.accumulate(cum)
            drawdowns = cum - rolling_max
            max_dd = drawdowns.min()
            peak = rolling_max.max()
            if peak != 0:
                max_drawdown = round(float(max_dd / max(abs(peak), 1) * 100), 2)
    except Exception:
        pass

    return {
        "total_trades": total_trades,
        "avg_pnl": float(row.avg_pnl or 0),
        "total_pnl": float(row.total_pnl or 0),
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


@router.get("/daily-pnl")
async def get_daily_pnl(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Daily P&L breakdown for desk headers and charts."""
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return {"series": [], "total_pnl": 0.0, "today_pnl": 0.0}

    since = datetime.now(timezone.utc) - timedelta(days=days)

    result = await db.execute(
        select(
            func.date_trunc("day", Trade.closed_at).label("day"),
            func.sum(Trade.realized_pnl).label("daily_pnl"),
            func.count(Trade.id).label("n_trades"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= since,
            Trade.realized_pnl.isnot(None),
        )
        .group_by(func.date_trunc("day", Trade.closed_at))
        .order_by(func.date_trunc("day", Trade.closed_at))
    )
    rows = result.all()

    if not rows:
        return {"series": [], "total_pnl": 0.0, "today_pnl": 0.0}

    today_str = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
    total_pnl = 0.0
    today_pnl = 0.0
    series = []

    for row in rows:
        day_val = row.day
        day_str = (
            day_val.strftime("%Y-%m-%d")
            if hasattr(day_val, "strftime")
            else str(day_val)[:10]
        )
        pnl = float(row.daily_pnl or 0)
        total_pnl += pnl
        if day_str == today_str:
            today_pnl = pnl
        series.append(
            {"date": day_str, "pnl": round(pnl, 2), "trades": row.n_trades}
        )

    return {"series": series, "total_pnl": round(total_pnl, 2), "today_pnl": round(today_pnl, 2)}