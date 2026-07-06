"""Analytics and performance metrics endpoints."""
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import settings
from app.database import get_db
from app.models.account import Account
from app.models.position import Position
from app.models.order import Order
from app.models.slippage import SlippageRecord
from app.models.trade import Trade
from app.models.user import User
from app.utils.logging import logger

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Simple in‑memory cache for expensive queries (per‑user, short TTL)
_PERFORMANCE_CACHE: Dict[str, Dict[str, Any]] = {}
_DAILY_PNL_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _cache_get(cache: Dict[str, Dict[str, Any]], key: str) -> Any | None:
    entry = cache.get(key)
    if entry and (datetime.now(timezone.utc) - entry["ts"]).total_seconds() < _CACHE_TTL_SECONDS:
        return entry["value"]
    # stale or missing
    cache.pop(key, None)
    return None


def _cache_set(cache: Dict[str, Dict[str, Any]], key: str, value: Any) -> None:
    cache[key] = {"value": value, "ts": datetime.now(timezone.utc)}


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
        from sqlalchemy import desc

        # Return the most recent price snapshots per symbol/exchange for comparison
        result = await db.execute(
            select(OHLCV.symbol, OHLCV.exchange, OHLCV.close, OHLCV.ts)
            .order_by(desc(OHLCV.ts))
            .limit(200)
        )
        rows = result.all()

        # Group by symbol to find cross‑exchange spreads
        by_symbol: defaultdict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_symbol[row.symbol].append(
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
            spread_pct = (spread / min(prices) * 100) if min(prices) > 0 else 0.0
            if spread_pct > 0.05:  # only surface if >5bps spread
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


async def _user_account_ids(db: AsyncSession, user_id: str) -> List[str]:
    """Return all account IDs owned by the given user. Used to scope queries."""
    result = await db.execute(select(Account.id).where(Account.user_id == user_id))
    return [row[0] for row in result.all()]


@router.get("/performance")
async def get_performance(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Aggregate trade performance stats — scoped to current user's accounts."""
    cache_key = f"performance:{current_user.id}"
    cached = _cache_get(_PERFORMANCE_CACHE, cache_key)
    if cached:
        return cached

    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        empty_result = {
            "total_trades": 0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": None,
            "max_drawdown": None,
        }
        _cache_set(_PERFORMANCE_CACHE, cache_key, empty_result)
        return empty_result

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

    # Early exit if no trades – no need for daily aggregation
    if total_trades == 0:
        perf = {
            "total_trades": 0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": None,
            "max_drawdown": None,
        }
        _cache_set(_PERFORMANCE_CACHE, cache_key, perf)
        return perf

    # Compute Sharpe and max drawdown from daily PnL series
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
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
            daily_pnls = [float(r.daily_pnl) for r in daily_rows]
            series = pd.Series(daily_pnls)
            mean_r = series.mean()
            std_r = series.std()
            if std_r > 0:
                sharpe_ratio = round(float(mean_r / std_r * (252 ** 0.5)), 4)
            cum = series.cumsum()
            rolling_max = cum.cummax()
            drawdown = cum - rolling_max
            if not drawdown.empty:
                max_dd = drawdown.min()
                peak = max(abs(float(cum.max())), 1)
                max_drawdown = round(float(max_dd / peak * 100), 2)
    except Exception as exc:
        logger.debug("Failed to compute sharpe/drawdown", error=str(exc))

    perf = {
        "total_trades": total_trades,
        "avg_pnl": float(row.avg_pnl or 0),
        "total_pnl": float(row.total_pnl or 0),
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }
    _cache_set(_PERFORMANCE_CACHE, cache_key, perf)
    return perf


@router.get("/daily-pnl")
async def get_daily_pnl(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Daily P&L breakdown for desk headers and charts."""
    cache_key = f"daily_pnl:{current_user.id}:{days}"
    cached = _cache_get(_DAILY_PNL_CACHE, cache_key)
    if cached:
        return cached

    account_ids = await _user_account_ids(db, current_user.id)
    since = datetime.now(timezone.utc) - timedelta(days=days)

    if not account_ids:
        empty = {"series": [], "total_pnl": 0.0, "today_pnl": 0.0}
        _cache_set(_DAILY_PNL_CACHE, cache_key, empty)
        return empty

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

    today_str = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
    total_pnl = 0.0
    today_pnl = 0.0
    series: List[Dict[str, Any]] = []

    for row in rows:
        day_val = row.day
        day_str = (
            day_val.strftime("%Y-%m-%d")
            if hasattr(day_val, "strftime")
            else str(day_val)[:10]
        )
        pnl = float(row.daily_pnl or 0)
        total_pnl += pnl
        series.append(
            {"date": day_str, "pnl": round(pnl, 2), "trades": int(row.n_trades or 0)}
        )
        if day_str == today_str:
            today_pnl = pnl

    response = {"series": series, "total_pnl": round(total_pnl, 2), "today_pnl": round(today_pnl, 2)}
    _cache_set(_DAILY_PNL_CACHE, cache_key, response)
    return response