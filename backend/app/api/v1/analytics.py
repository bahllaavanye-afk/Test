"""Analytics and performance metrics endpoints."""
import math
import re
from datetime import UTC, date, datetime, timedelta

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import settings
from app.database import get_db
from app.models.account import Account
from app.models.order import Order
from app.models.position import Position
from app.models.slippage import SlippageRecord
from app.models.strategy import Strategy
from app.models.trade import Trade
from app.models.user import User
from app.utils.logging import logger

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/")
async def analytics_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """High-level analytics summary: available modules and quick stats."""
    try:
        trade_count_result = await db.execute(
            select(func.count()).select_from(Trade)
        )
        trade_count = trade_count_result.scalar() or 0
    except Exception:
        trade_count = 0
    return {
        "modules": [
            "arb-opportunities", "performance", "slippage", "attribution",
            "macro", "sentiment", "correlation", "tearsheet", "equity-curve",
            "monthly-returns", "portfolio-greeks",
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

    Reads from the OHLCV table looking for cross-exchange price discrepancies.
    Returns an empty list when no data is available rather than 404.
    """
    try:
        from sqlalchemy import desc

        from app.models.market_data import OHLCV

        # Return the most recent price snapshots per symbol/exchange for comparison
        result = await db.execute(
            select(OHLCV.symbol, OHLCV.exchange, OHLCV.close, OHLCV.ts)
            .order_by(desc(OHLCV.ts))
            .limit(200)
        )
        rows = result.all()

        # Group by symbol to find cross-exchange spreads
        from collections import defaultdict
        by_symbol: dict = defaultdict(list)
        for row in rows:
            by_symbol[row.symbol].append({
                "exchange": row.exchange,
                "price": float(row.close),
                "ts": row.ts.isoformat() if row.ts else None,
            })

        opportunities = []
        for symbol, entries in by_symbol.items():
            if len(entries) < 2:
                continue
            prices = [e["price"] for e in entries if e["price"] > 0]
            if len(prices) < 2:
                continue
            spread = max(prices) - min(prices)
            spread_pct = spread / min(prices) * 100 if min(prices) > 0 else 0.0
            if spread_pct > 0.05:  # only surface if >5bps spread
                opportunities.append({
                    "symbol": symbol,
                    "spread": round(spread, 6),
                    "spread_pct": round(spread_pct, 4),
                    "exchanges": entries,
                })

        return opportunities

    except Exception as exc:
        logger.warning("arb-opportunities endpoint failed", error=str(exc))
        return []


async def _user_account_ids(db: AsyncSession, user_id: str) -> list[str]:
    """Return all account IDs owned by the given user. Used to scope queries."""
    result = await db.execute(
        select(Account.id).where(Account.user_id == user_id)
    )
    return [row[0] for row in result.all()]


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
                Trade.closed_at >= datetime.now(UTC) - timedelta(days=365),
                Trade.realized_pnl.isnot(None),
            )
            .group_by(func.date_trunc("day", Trade.closed_at))
            .order_by(func.date_trunc("day", Trade.closed_at))
        )
        daily_rows = daily_result.all()
        if len(daily_rows) >= 5:
            daily_pnls = [float(r.daily_pnl) for r in daily_rows]
            s = pd.Series(daily_pnls)
            mean_r = s.mean()
            std_r = s.std()
            if std_r > 0:
                sharpe_ratio = round(float(mean_r / std_r * (252 ** 0.5)), 4)
            cum = s.cumsum()
            rolling_max = cum.cummax()
            dd = (cum - rolling_max)
            max_drawdown = round(float(dd.min() / max(abs(float(cum.max())), 1) * 100), 2) if len(dd) > 0 else 0.0
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
    since = datetime.now(UTC) - timedelta(days=days)

    if not account_ids:
        return {"series": [], "total_pnl": 0.0, "today_pnl": 0.0}

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

    today = datetime.now(UTC).date()
    today_pnl = 0.0
    total_pnl = 0.0
    series = []
    for row in rows:
        day_val = row.day
        day_str = day_val.strftime("%Y-%m-%d") if hasattr(day_val, "strftime") else str(day_val)[:10]
        pnl = float(row.daily_pnl or 0)
        total_pnl += pnl
        series.append({"date": day_str, "pnl": round(pnl, 2), "trades": row.n_trades})
        if day_str == today.strftime("%Y-%m-%d"):
            today_pnl = pnl

    return {"series": series, "total_pnl": round(total_pnl, 2), "today_pnl": round(today_pnl, 2)}


@router.get("/slippage")
async def get_slippage_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Average slippage by execution algorithm — scoped to current user's orders."""
    from app.models.order import Order as OrderModel
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return []
    result = await db.execute(
        select(
            SlippageRecord.execution_algo,
            func.avg(SlippageRecord.slippage_bps).label("avg_bps"),
            func.count(SlippageRecord.id).label("count"),
        )
        .join(OrderModel, SlippageRecord.order_id == OrderModel.id)
        .where(OrderModel.account_id.in_(account_ids))
        .group_by(SlippageRecord.execution_algo)
    )
    rows = result.all()
    return [{"algo": r.execution_algo, "avg_bps": round(float(r.avg_bps or 0), 2), "count": r.count} for r in rows]


@router.get("/tca")
async def get_tca(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Transaction Cost Analysis — execution quality metrics."""
    since = datetime.now(UTC) - timedelta(days=days)
    account_ids = await _user_account_ids(db, current_user.id)

    # Count total trades in period
    trade_count_result = await db.execute(
        select(func.count(Trade.id)).where(
            Trade.account_id.in_(account_ids) if account_ids else Trade.account_id.isnot(None),
            Trade.closed_at >= since,
        )
    )
    total_trades = int(trade_count_result.scalar() or 0)

    # Check if SlippageRecord data is available for this user's accounts
    slippage_data_available = False
    avg_slippage_bps = None
    median_slippage_bps = None
    total_estimated_cost_usd = None
    by_strategy_slippage: list[dict] = []
    by_execution_algo: list[dict] = []
    by_hour_of_day: list[dict] = []
    best_strategy_for_execution = None
    worst_strategy_for_execution = None

    try:
        from app.models.order import Order as OrderModel
        slippage_count_result = await db.execute(
            select(func.count(SlippageRecord.id))
            .join(OrderModel, SlippageRecord.order_id == OrderModel.id)
            .where(
                OrderModel.account_id.in_(account_ids) if account_ids else OrderModel.account_id.isnot(None),
                SlippageRecord.created_at >= since,
            )
        )
        slippage_count = int(slippage_count_result.scalar() or 0)

        if slippage_count > 0:
            slippage_data_available = True

            # Average and median slippage bps
            agg_result = await db.execute(
                select(
                    func.avg(SlippageRecord.slippage_bps).label("avg_bps"),
                    func.percentile_cont(0.5).within_group(
                        SlippageRecord.slippage_bps
                    ).label("median_bps"),
                    func.sum(
                        case((SlippageRecord.slippage_bps.isnot(None),
                              SlippageRecord.slippage_bps * SlippageRecord.fill_price / 10000), else_=0)
                    ).label("total_cost"),
                )
                .join(OrderModel, SlippageRecord.order_id == OrderModel.id)
                .where(
                    OrderModel.account_id.in_(account_ids) if account_ids else OrderModel.account_id.isnot(None),
                    SlippageRecord.created_at >= since,
                )
            )
            agg_row = agg_result.one()
            avg_slippage_bps = round(float(agg_row.avg_bps or 0), 4) if agg_row.avg_bps is not None else None
            median_slippage_bps = round(float(agg_row.median_bps or 0), 4) if agg_row.median_bps is not None else None
            total_estimated_cost_usd = round(float(agg_row.total_cost or 0), 2) if agg_row.total_cost is not None else None

            # By strategy via Order → Strategy FK (avoids Cartesian product through account_id)
            strat_result = await db.execute(
                select(
                    func.coalesce(Strategy.name, "manual").label("strategy_name"),
                    func.avg(SlippageRecord.slippage_bps).label("avg_bps"),
                    func.count(SlippageRecord.id).label("n_trades"),
                )
                .join(OrderModel, SlippageRecord.order_id == OrderModel.id)
                .outerjoin(Strategy, Strategy.id == OrderModel.strategy_id)
                .where(
                    OrderModel.account_id.in_(account_ids) if account_ids else OrderModel.account_id.isnot(None),
                    SlippageRecord.created_at >= since,
                )
                .group_by(Strategy.name)
                .order_by(func.avg(SlippageRecord.slippage_bps))
            )
            strat_rows = strat_result.all()
            by_strategy_slippage = [
                {
                    "strategy": r.strategy_name or "manual",
                    "avg_slippage_bps": round(float(r.avg_bps or 0), 4),
                    "num_trades": r.n_trades,
                }
                for r in strat_rows
            ]
            if by_strategy_slippage:
                best_strategy_for_execution = by_strategy_slippage[0]["strategy"]
                worst_strategy_for_execution = by_strategy_slippage[-1]["strategy"]

            # By execution algo
            algo_result = await db.execute(
                select(
                    SlippageRecord.execution_algo,
                    func.avg(SlippageRecord.slippage_bps).label("avg_bps"),
                )
                .join(OrderModel, SlippageRecord.order_id == OrderModel.id)
                .where(
                    OrderModel.account_id.in_(account_ids) if account_ids else OrderModel.account_id.isnot(None),
                    SlippageRecord.created_at >= since,
                )
                .group_by(SlippageRecord.execution_algo)
                .order_by(func.avg(SlippageRecord.slippage_bps))
            )
            algo_rows = algo_result.all()
            by_execution_algo = [
                {
                    "algo": r.execution_algo or "unknown",
                    "avg_slippage_bps": round(float(r.avg_bps or 0), 4),
                }
                for r in algo_rows
            ]

            # By hour of day (UTC)
            hour_result = await db.execute(
                select(
                    func.extract("hour", SlippageRecord.created_at).label("hour"),
                    func.avg(SlippageRecord.slippage_bps).label("avg_bps"),
                )
                .join(OrderModel, SlippageRecord.order_id == OrderModel.id)
                .where(
                    OrderModel.account_id.in_(account_ids) if account_ids else OrderModel.account_id.isnot(None),
                    SlippageRecord.created_at >= since,
                )
                .group_by(func.extract("hour", SlippageRecord.created_at))
                .order_by(func.extract("hour", SlippageRecord.created_at))
            )
            hour_rows = hour_result.all()
            by_hour_of_day = [
                {
                    "hour": int(r.hour),
                    "avg_slippage_bps": round(float(r.avg_bps or 0), 4),
                }
                for r in hour_rows
            ]
    except Exception as exc:
        logger.warning("TCA slippage aggregation failed", error=str(exc))

    data_source = (
        "slippage_records" if slippage_data_available else
        ("trade_estimates" if total_trades > 0 else "insufficient")
    )

    # Always compute Trade-level stats
    avg_pnl_per_trade = None
    win_rate = None
    profit_factor = None
    try:
        trade_stats_result = await db.execute(
            select(
                func.avg(Trade.realized_pnl).label("avg_pnl"),
                func.count(Trade.id).label("n_trades"),
                func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("wins"),
                func.sum(case((Trade.realized_pnl > 0, Trade.realized_pnl), else_=0)).label("gross_profit"),
                func.sum(case((Trade.realized_pnl <= 0, Trade.realized_pnl), else_=0)).label("gross_loss"),
            )
            .where(
                Trade.account_id.in_(account_ids) if account_ids else Trade.account_id.isnot(None),
                Trade.closed_at >= since,
                Trade.realized_pnl.isnot(None),
            )
        )
        ts_row = trade_stats_result.one()
        n_trades = int(ts_row.n_trades or 0)
        if n_trades > 0:
            avg_pnl_per_trade = round(float(ts_row.avg_pnl or 0), 4)
            wins = int(ts_row.wins or 0)
            win_rate = round(wins / n_trades, 4)
            gross_profit = float(ts_row.gross_profit or 0)
            gross_loss = float(ts_row.gross_loss or 0)
            if gross_loss < 0:
                profit_factor = round(gross_profit / abs(gross_loss), 4)
    except Exception as exc:
        logger.warning("TCA trade stats failed", error=str(exc))

    return {
        "period_days": days,
        "total_trades": total_trades,
        "data_source": data_source,
        "avg_slippage_bps": avg_slippage_bps,
        "median_slippage_bps": median_slippage_bps,
        "total_estimated_cost_usd": total_estimated_cost_usd,
        "by_strategy": by_strategy_slippage,
        "by_execution_algo": by_execution_algo,
        "by_hour_of_day": by_hour_of_day,
        "best_strategy_for_execution": best_strategy_for_execution,
        "worst_strategy_for_execution": worst_strategy_for_execution,
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


@router.get("/attribution")
async def get_pnl_attribution(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """PnL attribution breakdown by strategy, time-of-day, day-of-week."""
    since = datetime.now(UTC) - timedelta(days=days)
    account_ids = await _user_account_ids(db, current_user.id)

    where_clause = [
        Trade.closed_at >= since,
        Trade.realized_pnl.isnot(None),
    ]
    if account_ids:
        where_clause.append(Trade.account_id.in_(account_ids))

    # Overall totals
    totals_result = await db.execute(
        select(
            func.count(Trade.id).label("total_trades"),
            func.sum(Trade.realized_pnl).label("total_pnl"),
            func.avg(Trade.realized_pnl).label("avg_pnl"),
            func.sum(case((Trade.realized_pnl > 0, Trade.realized_pnl), else_=0)).label("gross_profit"),
            func.sum(case((Trade.realized_pnl <= 0, Trade.realized_pnl), else_=0)).label("gross_loss"),
        ).where(*where_clause)
    )
    totals_row = totals_result.one()
    total_trades = int(totals_row.total_trades or 0)
    total_pnl = float(totals_row.total_pnl or 0)
    avg_pnl = float(totals_row.avg_pnl or 0)
    gross_profit = float(totals_row.gross_profit or 0)
    gross_loss = float(totals_row.gross_loss or 0)
    profit_factor = round(gross_profit / abs(gross_loss), 4) if gross_loss < 0 else None

    # Per-strategy breakdown
    strat_result = await db.execute(
        select(
            Trade.strategy_name,
            func.count(Trade.id).label("n_trades"),
            func.sum(Trade.realized_pnl).label("total_pnl"),
            func.avg(Trade.realized_pnl).label("avg_pnl"),
            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("wins"),
            func.avg(Trade.realized_pnl * Trade.realized_pnl).label("avg_sq_pnl"),
        )
        .where(*where_clause)
        .group_by(Trade.strategy_name)
        .order_by(func.sum(Trade.realized_pnl).desc())
    )
    strat_rows = strat_result.all()

    by_strategy = []
    for r in strat_rows:
        s_trades = int(r.n_trades or 0)
        s_pnl = float(r.total_pnl or 0)
        s_avg = float(r.avg_pnl or 0)
        s_wins = int(r.wins or 0)
        s_avg_sq = float(r.avg_sq_pnl or 0)
        s_std = math.sqrt(max(s_avg_sq - s_avg * s_avg, 0.0))
        contribution_pct = round(s_pnl / total_pnl * 100, 2) if total_pnl != 0 else 0.0
        win_rate_s = round(s_wins / max(s_trades, 1), 4)
        sharpe_proxy = round(s_avg / s_std * math.sqrt(252), 4) if s_std > 0 else None
        by_strategy.append({
            "name": r.strategy_name or "manual",
            "total_pnl": round(s_pnl, 2),
            "contribution_pct": contribution_pct,
            "num_trades": s_trades,
            "win_rate": win_rate_s,
            "avg_pnl_per_trade": round(s_avg, 4),
            "sharpe_proxy": sharpe_proxy,
        })

    # Day-of-week breakdown (func.extract('dow', ...) → 0=Sun ... 6=Sat on PostgreSQL)
    _DOW_MAP = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
    dow_result = await db.execute(
        select(
            func.extract("dow", Trade.closed_at).label("dow"),
            func.avg(Trade.realized_pnl).label("avg_pnl"),
        )
        .where(*where_clause)
        .group_by(func.extract("dow", Trade.closed_at))
        .order_by(func.extract("dow", Trade.closed_at))
    )
    dow_rows = dow_result.all()
    by_day_of_week: dict[str, float] = {d: 0.0 for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}
    for r in dow_rows:
        day_name = _DOW_MAP.get(int(r.dow), "Mon")
        by_day_of_week[day_name] = round(float(r.avg_pnl or 0), 4)

    best_day = max(by_day_of_week, key=lambda k: by_day_of_week[k])
    worst_day = min(by_day_of_week, key=lambda k: by_day_of_week[k])

    # Hour-of-day breakdown (UTC)
    hour_result = await db.execute(
        select(
            func.extract("hour", Trade.closed_at).label("hour"),
            func.avg(Trade.realized_pnl).label("avg_pnl"),
        )
        .where(*where_clause)
        .group_by(func.extract("hour", Trade.closed_at))
        .order_by(func.extract("hour", Trade.closed_at))
    )
    hour_rows = hour_result.all()
    by_hour_of_day: dict[str, float] = {str(h): 0.0 for h in range(24)}
    for r in hour_rows:
        by_hour_of_day[str(int(r.hour))] = round(float(r.avg_pnl or 0), 4)

    best_hour_utc = max(range(24), key=lambda h: by_hour_of_day[str(h)])
    worst_hour_utc = min(range(24), key=lambda h: by_hour_of_day[str(h)])

    return {
        "period_days": days,
        "total_pnl": round(total_pnl, 2),
        "total_trades": total_trades,
        "expectancy_usd": round(avg_pnl, 4),
        "profit_factor": profit_factor,
        "by_strategy": by_strategy,
        "by_day_of_week": by_day_of_week,
        "by_hour_of_day": by_hour_of_day,
        "best_day": best_day,
        "worst_day": worst_day,
        "best_hour_utc": best_hour_utc,
        "worst_hour_utc": worst_hour_utc,
    }


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
    start_dt = (datetime.now(UTC) - timedelta(days=days + 10)).strftime("%Y-%m-%d")
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
    except Exception as exc:
        logger.warning("_fetch_alpaca_bars failed", error=str(exc))
    return prices


@router.get("/correlation")
async def get_correlation_matrix(
    account_id: str | None = Query(None),
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
    except Exception as exc:
        logger.warning("correlation matrix: failed to fetch position symbols", error=str(exc))
        symbols = []

    if not symbols:
        symbols = DEFAULT_SYMBOLS

    # Fetch price data from Alpaca (equity symbols only)
    prices_map = await _fetch_alpaca_bars(symbols, days)

    if not prices_map:
        return {
            "symbols": symbols,
            "matrix": [],
            "computed_at": datetime.now(UTC).isoformat(),
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
            "computed_at": datetime.now(UTC).isoformat(),
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
        "computed_at": datetime.now(UTC).isoformat(),
    }


# ─── Tax Lots ─────────────────────────────────────────────────────────────────

ALPACA_QUOTES_URL = "https://data.alpaca.markets/v2/stocks/quotes/latest"


async def _fetch_latest_price(symbol: str) -> float | None:
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
    except Exception as exc:
        logger.debug("_fetch_latest_price failed", symbol=symbol, error=str(exc))
    return None


@router.get("/tax-lots/{symbol}")
async def get_tax_lots(
    symbol: str,
    account_id: str | None = Query(None),
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

    now = datetime.now(UTC)
    result_lots = []
    for i, lot in enumerate(open_lots):
        qty = float(lot["qty"])
        cost = float(lot["cost"])
        acquired = lot["acquired_at"]
        # Ensure timezone-aware
        if acquired and acquired.tzinfo is None:
            acquired = acquired.replace(tzinfo=UTC)

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


# ─── Portfolio Greeks ─────────────────────────────────────────────────────────

_ALPACA_OPTIONS_BASE = "https://paper-api.alpaca.markets"

_OPTION_SYMBOL_RE = re.compile(
    r"^[A-Z]{1,6}\d{6}[CP]\d{8}$"
)


def _is_option_symbol(symbol: str) -> bool:
    """Return True if symbol looks like an OCC-formatted option symbol."""
    return bool(_OPTION_SYMBOL_RE.match(symbol.upper()))


async def _fetch_options_snapshots_for_symbols(symbols: list[str]) -> dict[str, dict]:
    """Fetch Alpaca options snapshots for a list of option symbols."""
    if not symbols:
        return {}
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        "accept": "application/json",
    }
    results: dict[str, dict] = {}
    BATCH = 50
    async with httpx.AsyncClient(timeout=20.0) as client:
        for i in range(0, len(symbols), BATCH):
            batch = symbols[i : i + BATCH]
            try:
                resp = await client.get(
                    f"{_ALPACA_OPTIONS_BASE}/v2/options/snapshots",
                    params={"symbols": ",".join(batch), "feed": "indicative"},
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results.update(data.get("snapshots") or {})
            except Exception as exc:
                logger.debug("options snapshots batch failed", error=str(exc))
    return results


async def _get_account_equity_for_user(
    account_id: str | None,
    current_user: "User",
    db: AsyncSession,
) -> float:
    """Sum equity across user accounts (or a specific account). Falls back to 0."""
    acct_q = select(Account).where(
        Account.user_id == current_user.id,
        Account.is_active == True,  # noqa: E712
    )
    if account_id:
        acct_q = acct_q.where(Account.id == account_id)
    acct_result = await db.execute(acct_q)
    accounts = acct_result.scalars().all()

    total_equity = 0.0
    for acct in accounts:
        if acct.broker == "alpaca" and acct.encrypted_key:
            try:
                from app.brokers.alpaca_orders import get_alpaca_account
                data = await get_alpaca_account(acct)
                total_equity += float(data.get("equity", 0))
            except Exception as exc:
                logger.warning("account equity fetch failed", account_id=acct.id, error=str(exc))
    return total_equity


@router.get("/portfolio-greeks")
async def get_portfolio_greeks(
    account_id: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Aggregate portfolio-level options Greeks across all open option positions.
    Returns net delta/gamma/theta/vega, targets, warnings, and per-position breakdown.
    """
    # Resolve accounts
    acct_q = select(Account).where(
        Account.user_id == current_user.id,
        Account.is_active == True,  # noqa: E712
    )
    if account_id:
        acct_q = acct_q.where(Account.id == account_id)
    acct_result = await db.execute(acct_q)
    accounts = acct_result.scalars().all()

    account_ids = [a.id for a in accounts]
    if not account_ids:
        raise HTTPException(status_code=404, detail="No active accounts found for this user.")

    # Fetch all open positions
    pos_result = await db.execute(
        select(Position).where(Position.account_id.in_(account_ids))
    )
    all_positions = pos_result.scalars().all()

    # Filter to option positions
    option_positions = [p for p in all_positions if _is_option_symbol(p.symbol)]

    # Fetch account equity
    account_equity = await _get_account_equity_for_user(account_id, current_user, db)

    if not option_positions:
        theta_target = account_equity * 0.0015
        delta_limit = 0.30 * account_equity / 100.0
        return {
            "net_delta": 0.0,
            "net_gamma": 0.0,
            "net_theta": 0.0,
            "net_vega": 0.0,
            "theta_target": round(theta_target, 2),
            "theta_pct_of_target": 0.0,
            "delta_limit": round(delta_limit, 2),
            "is_delta_neutral": True,
            "warnings": [],
            "position_count": 0,
            "options_positions": [],
            "account_equity": round(account_equity, 2),
            "computed_at": datetime.now(UTC).isoformat(),
        }

    # Fetch snapshots for all option symbols
    opt_symbols = [p.symbol.upper() for p in option_positions]
    snapshots = await _fetch_options_snapshots_for_symbols(opt_symbols)

    # Aggregate Greeks
    net_delta = 0.0
    net_gamma = 0.0
    net_theta = 0.0
    net_vega = 0.0
    positions_out = []

    for pos in option_positions:
        sym = pos.symbol.upper()
        qty = float(pos.quantity)
        snap = snapshots.get(sym, {})
        greeks = snap.get("greeks") or {}
        iv = snap.get("impliedVolatility")

        delta = greeks.get("delta") or 0.0
        gamma = greeks.get("gamma") or 0.0
        theta = greeks.get("theta") or 0.0
        vega = greeks.get("vega") or 0.0

        # Multiply by quantity and 100 (contract multiplier)
        multiplier = qty * 100.0
        pos_delta = delta * multiplier
        pos_gamma = gamma * multiplier
        pos_theta = theta * multiplier
        pos_vega = vega * multiplier

        net_delta += pos_delta
        net_gamma += pos_gamma
        net_theta += pos_theta
        net_vega += pos_vega

        positions_out.append({
            "symbol": sym,
            "quantity": qty,
            "delta": round(delta, 4),
            "gamma": round(gamma, 4),
            "theta": round(theta, 4),
            "vega": round(vega, 4),
            "iv": round(float(iv), 4) if iv is not None else None,
            "position_delta": round(pos_delta, 4),
            "position_gamma": round(pos_gamma, 4),
            "position_theta": round(pos_theta, 4),
            "position_vega": round(pos_vega, 4),
        })

    # Calculate targets
    theta_target = account_equity * 0.0015 if account_equity > 0 else 0.0
    delta_limit = 0.30 * account_equity / 100.0 if account_equity > 0 else 0.0
    theta_pct_of_target = (net_theta / theta_target * 100.0) if theta_target > 0 else 0.0
    is_delta_neutral = abs(net_delta) < delta_limit if delta_limit > 0 else True

    # Build warnings
    warnings = []
    if net_vega < -1000:
        warnings.append("Net vega exceeds -1000 — high IV risk")
    if not is_delta_neutral:
        warnings.append(f"Net delta ({net_delta:+.2f}) exceeds limit (±{delta_limit:.2f}) — portfolio not delta neutral")
    if theta_target > 0 and net_theta < theta_target * 0.5:
        warnings.append(f"Net theta (${net_theta:.2f}) is below 50% of target (${theta_target:.2f}) — consider adding premium")
    if account_equity == 0:
        warnings.append("Unable to fetch account equity — targets may be zero")

    return {
        "net_delta": round(net_delta, 4),
        "net_gamma": round(net_gamma, 4),
        "net_theta": round(net_theta, 4),
        "net_vega": round(net_vega, 4),
        "theta_target": round(theta_target, 2),
        "theta_pct_of_target": round(theta_pct_of_target, 2),
        "delta_limit": round(delta_limit, 2),
        "is_delta_neutral": is_delta_neutral,
        "warnings": warnings,
        "position_count": len(option_positions),
        "options_positions": positions_out,
        "account_equity": round(account_equity, 2),
        "computed_at": datetime.now(UTC).isoformat(),
    }


@router.get("/portfolio/snapshot")
async def get_portfolio_snapshot(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Aggregate portfolio KPIs for the dashboard snapshot widget — scoped to user."""
    account_ids = await _user_account_ids(db, current_user.id)

    today = datetime.now(UTC).date()
    yesterday = today - timedelta(days=1)

    if not account_ids:
        return {
            "total_pnl": 0.0, "today_pnl": 0.0, "today_pnl_trend": 0.0,
            "sharpe": 0.0, "win_rate": 0.0, "max_drawdown": 0.0, "open_positions": 0,
        }

    # All-time realized PnL (scoped to user's accounts)
    total_pnl_result = await db.execute(
        select(func.coalesce(func.sum(Trade.realized_pnl), 0.0))
        .where(Trade.account_id.in_(account_ids))
    )
    total_pnl = float(total_pnl_result.scalar_one())

    # Today's realized PnL
    today_pnl_result = await db.execute(
        select(func.coalesce(func.sum(Trade.realized_pnl), 0.0)).where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= datetime.combine(today, datetime.min.time()).replace(tzinfo=UTC),
        )
    )
    today_pnl = float(today_pnl_result.scalar_one())

    # Yesterday's realized PnL (for trend)
    yesterday_pnl_result = await db.execute(
        select(func.coalesce(func.sum(Trade.realized_pnl), 0.0)).where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= datetime.combine(yesterday, datetime.min.time()).replace(tzinfo=UTC),
            Trade.closed_at < datetime.combine(today, datetime.min.time()).replace(tzinfo=UTC),
        )
    )
    yesterday_pnl = float(yesterday_pnl_result.scalar_one())

    # Win rate: positive trades / total trades (last 90 days)
    since_90 = datetime.now(UTC) - timedelta(days=90)
    wins_result = await db.execute(
        select(
            func.count(Trade.id).label("total"),
            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("wins"),
        ).where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= since_90,
        )
    )
    wins_row = wins_result.one()
    total_trades = wins_row.total or 0
    wins = int(wins_row.wins or 0)
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # Open positions count (scoped to user)
    open_pos_result = await db.execute(
        select(func.count(Position.id)).where(Position.account_id.in_(account_ids))
    )
    open_positions = int(open_pos_result.scalar_one() or 0)

    # Sharpe ratio: annualized from last 252 daily PnL values (scoped)
    daily_pnl_result = await db.execute(
        select(
            func.date_trunc("day", Trade.closed_at).label("day"),
            func.sum(Trade.realized_pnl).label("daily_pnl"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= datetime.now(UTC) - timedelta(days=365),
        )
        .group_by(func.date_trunc("day", Trade.closed_at))
        .order_by(func.date_trunc("day", Trade.closed_at))
    )
    daily_rows = daily_pnl_result.all()
    sharpe = 0.0
    max_drawdown = 0.0
    if len(daily_rows) >= 5:
        daily_pnls = [float(r.daily_pnl) for r in daily_rows]
        s = pd.Series(daily_pnls)
        mean = s.mean()
        std = s.std()
        sharpe = (mean / std * (252 ** 0.5)) if std > 0 else 0.0
        # Max drawdown from cumulative PnL
        cum = s.cumsum()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max)
        max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    return {
        "total_pnl": round(total_pnl, 2),
        "today_pnl": round(today_pnl, 2),
        "today_pnl_trend": round(today_pnl - yesterday_pnl, 2),
        "sharpe": round(sharpe, 4),
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_drawdown, 2),
        "open_positions": open_positions,
    }


@router.get("/equity-curve")
async def get_equity_curve(
    days: int = Query(365, ge=30, le=730, description="Lookback window in days"),
    initial_equity: float = Query(100_000.0, ge=1_000, description="Baseline equity to build curve from"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Daily cumulative equity curve built from realized trade P&L.

    Returns [{date, equity}] sorted ascending. The curve starts at
    initial_equity and adds each day's realized P&L going forward.
    Returns an empty list when there are no closed trades.
    """
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return []

    since = datetime.now(UTC) - timedelta(days=days)

    result = await db.execute(
        select(
            func.date_trunc("day", Trade.closed_at).label("day"),
            func.sum(Trade.realized_pnl).label("daily_pnl"),
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
        return []

    equity = initial_equity
    curve = []
    for row in rows:
        equity += float(row.daily_pnl)
        day = row.day
        curve.append({
            "date": day.strftime("%Y-%m-%d") if hasattr(day, "strftime") else str(day)[:10],
            "equity": round(equity, 2),
        })
    return curve


@router.get("/monthly-returns")
async def get_monthly_returns(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Monthly realized P&L as a percentage of a $100k baseline.

    Returns [{month: "Jan 2024", ret: 2.3}] for the last 24 months,
    sorted oldest-first. Used to populate the monthly return heatmap.
    Returns an empty list when there are no closed trades.
    """
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return []

    since = datetime.now(UTC) - timedelta(days=730)

    result = await db.execute(
        select(
            func.date_trunc("month", Trade.closed_at).label("month"),
            func.sum(Trade.realized_pnl).label("monthly_pnl"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= since,
            Trade.realized_pnl.isnot(None),
        )
        .group_by(func.date_trunc("month", Trade.closed_at))
        .order_by(func.date_trunc("month", Trade.closed_at))
    )
    rows = result.all()
    if not rows:
        return []

    # Build running equity to compute return % relative to start-of-month equity
    baseline = 100_000.0
    running_equity = baseline
    out = []
    for row in rows:
        monthly_pnl = float(row.monthly_pnl)
        ret_pct = round(monthly_pnl / max(running_equity, 1) * 100, 2)
        running_equity += monthly_pnl
        month = row.month
        out.append({
            "month": month.strftime("%b %Y") if hasattr(month, "strftime") else str(month)[:7],
            "ret": ret_pct,
        })
    return out


@router.get("/tearsheet")
async def get_tearsheet(
    days: int = Query(365, ge=90, le=730, description="Lookback window in days"),
    initial_equity: float = Query(100_000.0, ge=1_000, description="Baseline equity"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fund-style tearsheet metrics for investor pitch.

    Computes full performance analytics from Trade records:
      - Sharpe, Sortino, Calmar, Omega ratio, Ulcer Index
      - Total return, annualised return, max drawdown
      - Win rate, profit factor, avg win/loss
      - Benchmark comparison vs SPY (via yfinance)
      - Monthly returns heatmap data
      - Equity curve and drawdown curve

    Returns 404 when no trade data exists (no mock data).
    """
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        raise HTTPException(status_code=404, detail="No accounts found")

    since = datetime.now(UTC) - timedelta(days=days)

    # Fetch daily P&L
    result = await db.execute(
        select(
            func.date_trunc("day", Trade.closed_at).label("day"),
            func.sum(Trade.realized_pnl).label("daily_pnl"),
            func.count(Trade.id).label("n_trades"),
            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("n_wins"),
            func.avg(case((Trade.realized_pnl > 0, Trade.realized_pnl), else_=None)).label("avg_win"),
            func.avg(case((Trade.realized_pnl <= 0, Trade.realized_pnl), else_=None)).label("avg_loss"),
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
        raise HTTPException(status_code=404, detail="No trade data found in the requested period")

    # Build daily series
    daily_pnls = [float(r.daily_pnl) for r in rows]
    n_trades_total = sum(r.n_trades for r in rows)
    n_wins_total = sum(r.n_wins for r in rows)
    avg_win = float(next((r.avg_win for r in rows if r.avg_win is not None), 0) or 0)
    avg_loss = float(next((r.avg_loss for r in rows if r.avg_loss is not None), 0) or 0)

    s = pd.Series(daily_pnls)
    rf_daily = 0.05 / 252

    # Equity curve
    equity = initial_equity
    equity_curve = []
    drawdown_curve = []
    peak = initial_equity
    for i, (row, pnl) in enumerate(zip(rows, daily_pnls)):
        equity += pnl
        if equity > peak:
            peak = equity
        dd_pct = round((equity - peak) / peak * 100, 4) if peak > 0 else 0.0
        day = row.day
        day_str = day.strftime("%Y-%m-%d") if hasattr(day, "strftime") else str(day)[:10]
        equity_curve.append({"date": day_str, "equity": round(equity, 2)})
        drawdown_curve.append({"date": day_str, "drawdown_pct": dd_pct})

    # Performance metrics
    total_return = (equity - initial_equity) / initial_equity
    n_years = days / 365.0
    annualized_return = (1.0 + total_return) ** (1.0 / max(n_years, 0.01)) - 1.0
    max_dd = min(d["drawdown_pct"] for d in drawdown_curve) if drawdown_curve else 0.0

    daily_returns = s / initial_equity
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe = float((mean_ret - rf_daily) / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0

    downside_rets = daily_returns[daily_returns < rf_daily]
    downside_std = float(downside_rets.std()) if len(downside_rets) > 0 else 0.0
    sortino = float((mean_ret - rf_daily) / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

    calmar = float(annualized_return / abs(max_dd / 100.0)) if max_dd < 0 else 0.0

    # Omega ratio
    gains = daily_returns[daily_returns > rf_daily].sum()
    losses = abs(daily_returns[daily_returns <= rf_daily].sum())
    omega = float(gains / losses) if losses > 0 else float(gains > 0) * 999.0

    # Ulcer index
    drawdowns_pct = pd.Series([d["drawdown_pct"] for d in drawdown_curve])
    ulcer = float(math.sqrt((drawdowns_pct ** 2).mean())) if len(drawdowns_pct) > 0 else 0.0

    win_rate = n_wins_total / max(n_trades_total, 1)
    profit_factor = abs(avg_win * n_wins_total / (avg_loss * max(n_trades_total - n_wins_total, 1))) if avg_loss != 0 else 0.0

    # Monthly returns
    monthly_result = await db.execute(
        select(
            func.date_trunc("month", Trade.closed_at).label("month"),
            func.sum(Trade.realized_pnl).label("monthly_pnl"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= since,
            Trade.realized_pnl.isnot(None),
        )
        .group_by(func.date_trunc("month", Trade.closed_at))
        .order_by(func.date_trunc("month", Trade.closed_at))
    )
    monthly_rows = monthly_result.all()
    running_eq = initial_equity
    monthly_returns = []
    for row in monthly_rows:
        mpnl = float(row.monthly_pnl)
        ret_pct = round(mpnl / max(running_eq, 1) * 100, 2)
        running_eq += mpnl
        month = row.month
        monthly_returns.append({
            "month": month.strftime("%b %Y") if hasattr(month, "strftime") else str(month)[:7],
            "ret": ret_pct,
        })

    # Benchmark SPY via yfinance (best-effort, non-blocking)
    benchmark_sharpe_spy = None
    benchmark_return_spy = None
    try:
        import functools

        import yfinance as yf
        spy = await run_in_threadpool(functools.partial(yf.download, "SPY", period=f"{days}d", interval="1d", auto_adjust=True, progress=False))
        if spy is not None and len(spy) > 10:
            spy_close = spy["Close"].squeeze()
            spy_rets = spy_close.pct_change().dropna()
            spy_total = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1)
            spy_std = spy_rets.std()
            benchmark_sharpe_spy = round(
                float((spy_rets.mean() - rf_daily) / spy_std * math.sqrt(252)) if spy_std > 0 else 0.0, 4
            )
            benchmark_return_spy = round(spy_total * 100, 2)
    except Exception as exc:
        logger.warning("SPY benchmark fetch failed in tearsheet", error=str(exc))

    return {
        "period_days": days,
        "n_trading_days": len(rows),
        "n_trades": n_trades_total,
        # Core metrics
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "omega_ratio": round(omega, 4),
        "ulcer_index": round(ulcer, 4),
        # Returns
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(annualized_return * 100, 2),
        "max_drawdown_pct": round(max_dd, 2),
        # Trade stats
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "avg_win_pct": round(avg_win / initial_equity * 100, 4),
        "avg_loss_pct": round(avg_loss / initial_equity * 100, 4),
        # Benchmark
        "benchmark_sharpe_spy": benchmark_sharpe_spy,
        "benchmark_return_spy": benchmark_return_spy,
        # Time series
        "monthly_returns": monthly_returns,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "computed_at": datetime.now(UTC).isoformat(),
    }



@router.get("/live-stats")
async def get_live_stats(db: AsyncSession = Depends(get_db)):
    """Public endpoint for Landing page — real platform metrics.

    Returns null for performance metrics (Sharpe, win_rate, max_drawdown) when
    there is insufficient trade history rather than fabricated values.
    """
    from pathlib import Path as _Path
    _strategies_root = _Path(__file__).parents[3] / "app" / "strategies"
    _models_root = _Path(__file__).parents[3] / "app" / "ml" / "models"

    manual_files = list((_strategies_root / "manual").glob("*.py")) if (_strategies_root / "manual").exists() else []
    ml_files = list((_strategies_root / "ml_enhanced").glob("*.py")) if (_strategies_root / "ml_enhanced").exists() else []
    strategy_count = len([f for f in manual_files + ml_files if "__init__" not in f.name])

    model_files = list(_models_root.glob("*.py")) if _models_root.exists() else []
    model_count = len([f for f in model_files if "__init__" not in f.name and "base" not in f.name])

    # Compute real metrics from last-30d trade history
    sharpe_ratio = None
    max_drawdown_pct = None
    win_rate_pct = None
    total_trades = 0

    try:
        import math as _math

        from sqlalchemy import case, func, select

        from app.models.trade import Trade

        thirty_days_ago = datetime.now(UTC) - timedelta(days=30)

        # Total trade count + win rate
        stats = await db.execute(
            select(
                func.count(Trade.id).label("total"),
                func.sum(case((Trade.realized_pnl > 0, 1), else_=0)).label("wins"),
            ).where(Trade.closed_at >= thirty_days_ago)
        )
        stats_row = stats.first()
        total_trades = int(stats_row.total or 0)
        if total_trades >= 10:
            win_rate_pct = round(float(stats_row.wins or 0) / total_trades * 100, 1)

        # Daily PnL for Sharpe + max drawdown (requires >= 10 trading days)
        daily = await db.execute(
            select(
                func.date_trunc("day", Trade.closed_at).label("day"),
                func.sum(Trade.realized_pnl).label("daily_pnl"),
            )
            .where(Trade.closed_at >= thirty_days_ago)
            .group_by(func.date_trunc("day", Trade.closed_at))
            .order_by(func.date_trunc("day", Trade.closed_at))
        )
        daily_rows = daily.all()
        if len(daily_rows) >= 10:
            pnls = [float(r.daily_pnl or 0) for r in daily_rows]
            mean_d = sum(pnls) / len(pnls)
            variance = sum((x - mean_d) ** 2 for x in pnls) / len(pnls)
            std_d = variance ** 0.5
            sharpe_ratio = round((mean_d / std_d * _math.sqrt(252)) if std_d > 0 else 0.0, 2)

            # Max drawdown via running equity curve (start at 0)
            peak = 0.0
            equity = 0.0
            max_dd = 0.0
            for pnl in pnls:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (equity - peak) / max(abs(peak), 1e-9) * 100
                if dd < max_dd:
                    max_dd = dd
            max_drawdown_pct = round(max_dd, 1)
    except Exception:
        pass

    return {
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate_pct": win_rate_pct,
        "strategy_count": strategy_count,
        "model_count": model_count,
        "total_trades": total_trades,
    }


@router.get("/system-status")
async def get_system_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Aggregate system health status for the dashboard.

    Returns active strategy counts, last signal time, regime, VIX,
    open positions, today's P&L %, and strategies broken down by desk.
    """
    from sqlalchemy import desc

    from app.strategies import STRATEGY_REGISTRY

    account_ids = await _user_account_ids(db, current_user.id)

    # Strategy counts — use in-memory registry (DB strategies table may be empty on fresh deploy)
    strategy_classes = list(STRATEGY_REGISTRY.values())
    active_strategies = len(strategy_classes)  # all registered = active

    desk_map: dict[str, int] = {"equity": 0, "crypto": 0, "options": 0, "arbitrage": 0}
    for cls in strategy_classes:
        mt = getattr(cls, "market_type", "").lower()
        rb = getattr(cls, "risk_bucket", "").lower()
        if "arb" in rb or "arbitrage" in rb:
            desk_map["arbitrage"] += 1
        elif mt == "crypto":
            desk_map["crypto"] += 1
        elif mt in ("options", "option"):
            desk_map["options"] += 1
        else:
            desk_map["equity"] += 1
    total_strategies = len(strategy_classes)

    # Last signal time — most recent order created
    last_signal_at: str | None = None
    if account_ids:
        sig_result = await db.execute(
            select(Order.created_at)
            .where(Order.account_id.in_(account_ids))
            .order_by(desc(Order.created_at))
            .limit(1)
        )
        sig_row = sig_result.scalar_one_or_none()
        if sig_row:
            ts = sig_row
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            last_signal_at = ts.isoformat()

    # Open positions count
    open_positions = 0
    if account_ids:
        pos_result = await db.execute(
            select(func.count(Position.id)).where(Position.account_id.in_(account_ids))
        )
        open_positions = int(pos_result.scalar_one() or 0)

    # Today's P&L %
    today_pnl_pct = 0.0
    if account_ids:
        today = datetime.now(UTC).date()
        today_start = datetime.combine(today, datetime.min.time()).replace(tzinfo=UTC)
        today_pnl_result = await db.execute(
            select(func.coalesce(func.sum(Trade.realized_pnl), 0.0)).where(
                Trade.account_id.in_(account_ids),
                Trade.closed_at >= today_start,
            )
        )
        today_pnl = float(today_pnl_result.scalar_one())
        # Express as % of a $100k baseline
        today_pnl_pct = round(today_pnl / 100_000.0 * 100.0, 4)

    # Regime from macro snapshot (best-effort)
    regime: int | None = None
    vix: float | None = None
    try:
        from app.ml.features.macro_signals import get_macro_snapshot_cached
        macro = await get_macro_snapshot_cached()
        vix = macro.get("vix")
        bias = macro.get("macro_bias", "neutral")
        regime = 1 if bias == "risk_on" else (-1 if bias == "risk_off" else 0)
    except Exception:
        pass

    return {
        "active_strategies": active_strategies,
        "total_strategies": total_strategies,
        "last_signal_at": last_signal_at,
        "regime": regime,
        "vix": round(float(vix), 2) if vix is not None else None,
        "open_positions": open_positions,
        "today_pnl_pct": today_pnl_pct,
        "strategies_by_desk": desk_map,
    }


@router.get("/competition-report")
async def get_competition_report(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Compare QuantEdge performance vs major benchmarks and institutional funds.
    Returns live metrics + static reference benchmarks for investor pitch.
    """
    # Static reference benchmarks (long-run historical averages)
    BENCHMARKS = {
        "spy":       {"name": "S&P 500 (SPY)",          "sharpe": 0.47, "annual_return": 10.4, "max_dd": -57.0},
        "qqq":       {"name": "NASDAQ 100 (QQQ)",        "sharpe": 0.52, "annual_return": 14.2, "max_dd": -83.0},
        "brk_b":     {"name": "Warren Buffett (BRK-B)",  "sharpe": 0.79, "annual_return": 19.9, "max_dd": -48.0},
        "all_weather":{"name": "Ray Dalio All Weather",  "sharpe": 0.67, "annual_return": 8.2,  "max_dd": -20.0},
        "two_sigma": {"name": "Two Sigma (est.)",        "sharpe": 1.20, "annual_return": 20.0, "max_dd": -10.0},
        "renaissance":{"name": "Renaissance Medallion",  "sharpe": 2.10, "annual_return": 66.0, "max_dd": -5.0},
    }
    QUANTEDGE_TARGET = {"sharpe": 2.0, "annual_return": 25.0, "max_dd": -15.0}

    # Fetch live QuantEdge metrics from tearsheet (last 365 days)
    live_sharpe = None
    live_annual_return = None
    live_max_dd = None
    try:
        account_ids = await _user_account_ids(db, current_user.id)
        if account_ids:
            since = datetime.now(UTC) - timedelta(days=365)
            result = await db.execute(
                select(Trade.realized_pnl, Trade.exit_time)
                .where(Trade.account_id.in_(account_ids), Trade.exit_time >= since)
                .order_by(Trade.exit_time)
            )
            rows = result.all()
            if len(rows) >= 20:
                pnls = [float(r.realized_pnl or 0) for r in rows]
                arr = pd.Series(pnls)
                mean_r = arr.mean()
                std_r = arr.std()
                if std_r > 0:
                    live_sharpe = round(float(mean_r / std_r * math.sqrt(252)), 3)
                live_annual_return = round(float(arr.sum() / max(1, len(arr)) * 252 / 100), 1)
                running = arr.cumsum()
                peak = running.cummax()
                dd = (running - peak) / (peak.abs() + 1e-9) * 100
                live_max_dd = round(float(dd.min()), 1)
    except Exception:
        pass

    qs = live_sharpe or 0.0
    qr = live_annual_return or 0.0
    qd = live_max_dd or 0.0

    comparison = {}
    for key, bm in BENCHMARKS.items():
        comparison[key] = {
            **bm,
            "beating_sharpe": (qs > bm["sharpe"]) if qs else None,
            "beating_return": (qr > bm["annual_return"]) if qr else None,
            "sharpe_delta": round(qs - bm["sharpe"], 3) if qs else None,
            "return_delta": round(qr - bm["annual_return"], 1) if qr else None,
        }

    benchmarks_beaten = sum(1 for v in comparison.values() if v.get("beating_sharpe") is True)

    return {
        "quantedge": {
            "sharpe": qs,
            "annual_return_pct": qr,
            "max_drawdown_pct": qd,
            "data_available": live_sharpe is not None,
        },
        "target": QUANTEDGE_TARGET,
        "benchmarks": comparison,
        "benchmarks_beaten": benchmarks_beaten,
        "total_benchmarks": len(BENCHMARKS),
        "rank_summary": (
            f"Beating {benchmarks_beaten}/{len(BENCHMARKS)} benchmarks on Sharpe ratio"
            if live_sharpe else
            "Insufficient trade history — need ≥20 closed trades for comparison"
        ),
        "computed_at": datetime.now(UTC).isoformat(),
    }


@router.get("/is-analysis")
async def get_is_analysis(
    days: int = Query(90, ge=30, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Implementation Shortfall (IS) analysis — institutional-grade execution quality.

    IS = (fill_price - arrival_price) / arrival_price * 10_000 bps
    Positive IS = you paid more than the mid at order time (implementation cost).
    Negative IS = you bought cheaper than the mid (good execution).

    Returns:
      overall_is_bps: float — average IS across all tracked orders
      vwap_shortfall_bps: float — how far fills deviated from period VWAP
      by_strategy: list — IS breakdown per strategy name (joined via Order → Trade)
      by_algo: list — IS by execution algorithm (twap/vwap/limit_first/market)
      by_size_bucket: list — IS bucketed by order size (small/medium/large)
      execution_efficiency_score: float — 0-100 score (100=perfect, negative IS = >100)
      trend_7d: float — IS trend over last 7d vs prior 7d (improvement if negative)
      data_available: bool
    """
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return {
            "overall_is_bps": None,
            "vwap_shortfall_bps": None,
            "avg_execution_duration_seconds": None,
            "by_strategy": [],
            "by_algo": [],
            "by_size_bucket": [],
            "execution_efficiency_score": None,
            "trend_7d": None,
            "data_available": False,
        }

    cutoff = datetime.now(UTC) - timedelta(days=days)

    # Fetch SlippageRecords joined to Orders, filtered by user's accounts and date range
    stmt = (
        select(SlippageRecord)
        .join(Order, SlippageRecord.order_id == Order.id)
        .where(
            Order.account_id.in_(account_ids),
            SlippageRecord.created_at >= cutoff,
        )
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    if not records:
        return {
            "overall_is_bps": None,
            "vwap_shortfall_bps": None,
            "avg_execution_duration_seconds": None,
            "by_strategy": [],
            "by_algo": [],
            "by_size_bucket": [],
            "execution_efficiency_score": None,
            "trend_7d": None,
            "data_available": False,
        }

    # Helper: compute IS bps for a single record
    def _compute_is_bps(rec: SlippageRecord) -> float | None:
        if rec.is_cost_bps is not None:
            return float(rec.is_cost_bps)
        # Fallback: derive from fill_price and signal_price
        if rec.fill_price is not None and rec.signal_price is not None and float(rec.signal_price) != 0:
            return (float(rec.fill_price) - float(rec.signal_price)) / float(rec.signal_price) * 10_000
        return None

    # Helper: size bucket using signal_price as a proxy for notional
    def _size_bucket(rec: SlippageRecord) -> str:
        sp = float(rec.signal_price) if rec.signal_price is not None else 0.0
        if sp < 1_000:
            return "small"
        elif sp < 10_000:
            return "medium"
        else:
            return "large"

    # Collect all IS values
    is_values = [v for rec in records if (v := _compute_is_bps(rec)) is not None]
    vwap_values = [float(rec.vwap_shortfall_bps) for rec in records if rec.vwap_shortfall_bps is not None]
    duration_values = [float(rec.execution_duration_seconds) for rec in records if rec.execution_duration_seconds is not None]

    overall_is_bps = round(sum(is_values) / len(is_values), 4) if is_values else None
    avg_vwap_shortfall = round(sum(vwap_values) / len(vwap_values), 4) if vwap_values else None
    avg_duration = round(sum(duration_values) / len(duration_values), 2) if duration_values else None

    # by_algo breakdown
    algo_buckets: dict[str, list[float]] = {}
    for rec in records:
        algo = rec.execution_algo or "unknown"
        val = _compute_is_bps(rec)
        if val is not None:
            algo_buckets.setdefault(algo, []).append(val)

    by_algo = [
        {
            "algo": algo,
            "avg_is_bps": round(sum(vals) / len(vals), 4),
            "count": len(vals),
        }
        for algo, vals in sorted(algo_buckets.items())
    ]

    # by_strategy: join Order → strategy_id, then look up trade strategy_name
    # We fetch order_ids from our records and query trades matching account + strategy
    order_ids = [rec.order_id for rec in records]
    order_stmt = select(Order).where(Order.id.in_(order_ids))
    order_result = await db.execute(order_stmt)
    orders_map = {o.id: o for o in order_result.scalars().all()}

    # Gather strategy names from trades that share account_id (best-effort join)
    strategy_buckets: dict[str, list[float]] = {}
    for rec in records:
        val = _compute_is_bps(rec)
        if val is None:
            continue
        order = orders_map.get(rec.order_id)
        strategy_name: str | None = None
        if order is not None and order.strategy_id is not None:
            # Try to find a matching trade with same account and strategy
            trade_stmt = (
                select(Trade.strategy_name)
                .where(
                    Trade.account_id == order.account_id,
                    Trade.strategy_id == order.strategy_id,
                )
                .limit(1)
            )
            trade_result = await db.execute(trade_stmt)
            strategy_name = trade_result.scalar()
        strategy_name = strategy_name or (order.strategy_id if order else None) or "unknown"
        strategy_buckets.setdefault(strategy_name, []).append(val)

    by_strategy = [
        {
            "strategy": strat,
            "avg_is_bps": round(sum(vals) / len(vals), 4),
            "count": len(vals),
        }
        for strat, vals in sorted(strategy_buckets.items())
    ]

    # by_size_bucket
    size_buckets: dict[str, list[float]] = {}
    for rec in records:
        val = _compute_is_bps(rec)
        if val is None:
            continue
        bucket = _size_bucket(rec)
        size_buckets.setdefault(bucket, []).append(val)

    by_size_bucket = [
        {
            "bucket": bucket,
            "avg_is_bps": round(sum(vals) / len(vals), 4),
            "count": len(vals),
        }
        for bucket in ("small", "medium", "large")
        if bucket in size_buckets
        for vals in [size_buckets[bucket]]
    ]

    # Execution efficiency score: 100 = zero IS, >100 if negative IS (price improvement)
    execution_efficiency_score: float | None = None
    if overall_is_bps is not None:
        raw_score = 100.0 - max(0.0, overall_is_bps)
        execution_efficiency_score = round(max(0.0, raw_score), 2)

    # Trend: compare avg IS in last 7d vs prior 7d
    trend_7d: float | None = None
    now = datetime.now(UTC)
    recent_cutoff = now - timedelta(days=7)
    prior_cutoff = now - timedelta(days=14)

    recent_vals = [
        v for rec in records
        if rec.created_at >= recent_cutoff
        if (v := _compute_is_bps(rec)) is not None
    ]
    prior_vals = [
        v for rec in records
        if prior_cutoff <= rec.created_at < recent_cutoff
        if (v := _compute_is_bps(rec)) is not None
    ]

    if recent_vals and prior_vals:
        recent_avg = sum(recent_vals) / len(recent_vals)
        prior_avg = sum(prior_vals) / len(prior_vals)
        trend_7d = round(recent_avg - prior_avg, 4)  # negative = improvement

    return {
        "overall_is_bps": overall_is_bps,
        "vwap_shortfall_bps": avg_vwap_shortfall,
        "avg_execution_duration_seconds": avg_duration,
        "by_strategy": by_strategy,
        "by_algo": by_algo,
        "by_size_bucket": by_size_bucket,
        "execution_efficiency_score": execution_efficiency_score,
        "trend_7d": trend_7d,
        "data_available": True,
        "record_count": len(records),
        "computed_at": datetime.now(UTC).isoformat(),
    }


@router.get("/factor-attribution")
async def get_factor_attribution(
    days: int = Query(90, ge=30, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fama-French 5-factor attribution of realized P&L.

    Uses free yfinance data for factor proxies:
      - Market (MKT-RF): SPY daily returns - 0.04/252 (risk-free rate proxy)
      - Size (SMB): IWM - SPY (small minus big)
      - Value (HML): IVE - IVW (iShares S&P 500 Value vs Growth)
      - Momentum (MOM): MTUM - VTV (momentum ETF vs value)
      - Quality (QMJ): QUAL - USMV (quality vs min vol)

    Regresses daily strategy P&L (from Trade records) against these 5 factors
    using OLS (numpy lstsq). Returns factor loadings (betas), R-squared,
    alpha (Jensen's alpha annualized), and % of variance explained per factor.

    Returns:
      alpha_annualized_pct: float — Jensen's alpha annualized
      r_squared: float — goodness of fit (0-1)
      factors: dict — {factor_name: {beta, t_stat, contribution_pct}}
      total_explained_pct: float — % of P&L variance explained by the 5 factors
      unexplained_pct: float — alpha + noise
      data_start: str ISO date
      data_end: str ISO date
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not available"}

    import numpy as np

    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return {"error": "insufficient_data", "min_days_needed": 20}

    cutoff = datetime.now(UTC) - timedelta(days=days)

    # Fetch daily realized P&L grouped by date(closed_at)
    pnl_stmt = (
        select(
            func.date(Trade.closed_at).label("trade_date"),
            func.sum(Trade.realized_pnl).label("daily_pnl"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= cutoff,
        )
        .group_by(func.date(Trade.closed_at))
        .order_by(func.date(Trade.closed_at))
    )
    pnl_result = await db.execute(pnl_stmt)
    pnl_rows = pnl_result.all()

    if not pnl_rows or len(pnl_rows) < 20:
        return {"error": "insufficient_data", "min_days_needed": 20}

    # Build daily P&L series indexed by date
    pnl_by_date: dict[date, float] = {}
    for row in pnl_rows:
        trade_date = row.trade_date
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
        elif isinstance(trade_date, datetime):
            trade_date = trade_date.date()
        pnl_by_date[trade_date] = float(row.daily_pnl)

    # Download factor ETFs via yfinance (off event loop — blocking I/O)
    import functools
    etf_tickers = ["SPY", "IWM", "IVE", "IVW", "MTUM", "VTV", "QUAL", "USMV"]
    try:
        raw_df = await run_in_threadpool(functools.partial(yf.download, etf_tickers, period="1y", auto_adjust=True, progress=False))
        raw = raw_df["Close"]
        if hasattr(raw, "columns"):
            prices = raw
        else:
            return {"error": "yfinance returned unexpected data format"}
    except Exception as exc:
        logger.warning("yfinance download failed: %s", exc)
        return {"error": "factor_data_unavailable", "detail": str(exc)}

    # Compute daily factor returns
    prices = prices.dropna(how="all")
    if prices.empty or len(prices) < 5:
        return {"error": "insufficient_factor_data"}

    # Ensure all required tickers are present
    missing = [t for t in etf_tickers if t not in prices.columns]
    if missing:
        return {"error": "missing_factor_tickers", "missing": missing}

    pct = prices.pct_change().dropna()

    risk_free_daily = 0.04 / 252

    factor_series: dict[str, pd.Series] = {
        "MKT-RF": pct["SPY"] - risk_free_daily,
        "SMB":    pct["IWM"] - pct["SPY"],
        "HML":    pct["IVE"] - pct["IVW"],
        "MOM":    pct["MTUM"] - pct["VTV"],
        "QMJ":    pct["QUAL"] - pct["USMV"],
    }

    # Convert to DataFrame for easy alignment
    factor_df = pd.DataFrame(factor_series)
    factor_df.index = pd.to_datetime(factor_df.index).date  # type: ignore[assignment]

    # Build strategy P&L series aligned to factor dates (inner join)
    common_dates = sorted(set(factor_df.index) & set(pnl_by_date.keys()))

    if len(common_dates) < 20:
        return {"error": "insufficient_data", "min_days_needed": 20}

    y = np.array([pnl_by_date[d] for d in common_dates], dtype=float)
    X_factors = factor_df.loc[common_dates].values.astype(float)

    # Add constant (intercept) for alpha
    ones = np.ones((len(y), 1), dtype=float)
    X = np.hstack([ones, X_factors])  # shape (n, 6)

    # OLS: solve X @ beta = y
    coeffs, residuals_arr, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    alpha_daily = float(coeffs[0])
    betas = coeffs[1:]  # shape (5,)

    # Compute R-squared
    y_pred = X @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # Standard errors for t-stats
    n = len(y)
    k = X.shape[1]  # 6 (intercept + 5 factors)
    residuals = y - y_pred
    if n > k:
        sigma2 = float(np.sum(residuals ** 2)) / (n - k)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(sigma2 * np.diag(XtX_inv), 0.0))
        t_stats = coeffs / (se + 1e-12)
    else:
        se = np.zeros(k)
        t_stats = np.zeros(k)

    # Factor contribution: beta_i * std(factor_i) / std(y_pred) * 100
    factor_std = X_factors.std(axis=0)
    y_pred_std = float(y_pred.std()) if y_pred.std() > 1e-12 else 1.0
    contribution_pcts = [
        float(abs(betas[i]) * factor_std[i] / y_pred_std * 100)
        for i in range(5)
    ]
    total_explained = round(min(100.0, sum(contribution_pcts)), 2)

    # Annualize alpha: daily alpha * 252 * 100 (to percent)
    alpha_annualized_pct = round(alpha_daily * 252 * 100, 4)

    factor_names = list(factor_series.keys())
    factors_out = {
        factor_names[i]: {
            "beta": round(float(betas[i]), 6),
            "t_stat": round(float(t_stats[i + 1]), 4),  # +1 to skip intercept t_stat
            "contribution_pct": round(contribution_pcts[i], 2),
        }
        for i in range(5)
    }

    return {
        "alpha_annualized_pct": alpha_annualized_pct,
        "r_squared": round(r_squared, 6),
        "factors": factors_out,
        "total_explained_pct": total_explained,
        "unexplained_pct": round(max(0.0, 100.0 - total_explained), 2),
        "data_start": str(common_dates[0]),
        "data_end": str(common_dates[-1]),
        "n_observations": len(common_dates),
        "computed_at": datetime.now(UTC).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy leaderboard: ranks all strategies by live Sharpe, shows promotion
# stage and flags underperformers for auto-demotion.
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/strategy-leaderboard")
async def get_strategy_leaderboard(
    days: int = Query(90, ge=14, le=365),
    include_rejected: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Leaderboard of all strategies ranked by rolling Sharpe.
    Shows promotion stage, key metrics, and whether auto-demotion criteria are met.
    """
    from app.models.promotion import StrategyPromotion

    since = datetime.now(UTC) - timedelta(days=days)
    account_ids = await _user_account_ids(db, current_user.id)

    promo_filter = [StrategyPromotion.current_stage != "rejected"] if not include_rejected else []
    promo_result = await db.execute(
        select(StrategyPromotion).where(*promo_filter).order_by(StrategyPromotion.strategy_name)
    )
    promotions = promo_result.scalars().all()

    strategy_pnl: dict[str, list[float]] = {}
    if account_ids:
        trade_result = await db.execute(
            select(Trade.strategy_name, Trade.realized_pnl)
            .where(
                Trade.account_id.in_(account_ids),
                Trade.closed_at >= since,
                Trade.realized_pnl.isnot(None),
                Trade.strategy_name.isnot(None),
            )
        )
        for row in trade_result.all():
            strategy_pnl.setdefault(row.strategy_name, []).append(float(row.realized_pnl))

    rows = []
    for promo in promotions:
        pnls = strategy_pnl.get(promo.strategy_name, [])
        sharpe = 0.0
        win_rate = 0.0
        total_pnl = 0.0

        if len(pnls) >= 5:
            arr = pd.Series(pnls)
            std = arr.std()
            sharpe = round(float(arr.mean() / std * (252 ** 0.5)) if std > 1e-8 else 0.0, 3)
            win_rate = round(float((arr > 0).sum() / len(arr)), 3)
            total_pnl = round(float(arr.sum()), 2)

        if promo.current_stage == "live":
            stage_metrics = promo.live_metrics or {}
        elif promo.current_stage == "staging":
            stage_metrics = promo.staging_metrics or {}
        elif promo.current_stage == "shadow":
            stage_metrics = promo.shadow_metrics or {}
        else:
            stage_metrics = promo.paper_metrics or {}

        demote_flag = (
            promo.current_stage in ("live", "staging", "shadow")
            and sharpe < 0.3
            and len(pnls) >= 20
        )

        rows.append({
            "strategy_name": promo.strategy_name,
            "stage": promo.current_stage,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "num_trades": len(pnls),
            "promotion_ready": promo.promotion_ready,
            "promotion_ready_stage": promo.promotion_ready_stage,
            "demote_flag": demote_flag,
            "stage_days": stage_metrics.get("days_in_stage", 0),
            "stage_sharpe": stage_metrics.get("sharpe", 0.0),
        })

    stage_order = {"live": 0, "staging": 1, "shadow": 2, "paper": 3, "rejected": 4}
    rows.sort(key=lambda r: (stage_order.get(r["stage"], 5), -r["sharpe"]))

    demote_count = sum(1 for r in rows if r["demote_flag"])
    return {
        "leaderboard": rows,
        "total_strategies": len(rows),
        "demote_flag_count": demote_count,
        "period_days": days,
        "computed_at": datetime.now(UTC).isoformat(),
    }


@router.post("/strategy-leaderboard/auto-demote")
async def auto_demote_underperformers(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Auto-demote strategies flagged with demote_flag=True.
    Moves live → staging, staging → shadow, shadow → paper.
    """
    from app.models.promotion import StrategyPromotion

    since = datetime.now(UTC) - timedelta(days=90)
    account_ids = await _user_account_ids(db, current_user.id)

    strategy_pnl: dict[str, list[float]] = {}
    if account_ids:
        trade_result = await db.execute(
            select(Trade.strategy_name, Trade.realized_pnl)
            .where(
                Trade.account_id.in_(account_ids),
                Trade.closed_at >= since,
                Trade.realized_pnl.isnot(None),
                Trade.strategy_name.isnot(None),
            )
        )
        for row in trade_result.all():
            strategy_pnl.setdefault(row.strategy_name, []).append(float(row.realized_pnl))

    promo_result = await db.execute(
        select(StrategyPromotion).where(
            StrategyPromotion.current_stage.in_(["live", "staging", "shadow"])
        )
    )
    promotions = promo_result.scalars().all()

    demotion_map = {"live": "staging", "staging": "shadow", "shadow": "paper"}
    demoted = []

    for promo in promotions:
        pnls = strategy_pnl.get(promo.strategy_name, [])
        if len(pnls) < 20:
            continue
        arr = pd.Series(pnls)
        std = arr.std()
        sharpe = float(arr.mean() / std * (252 ** 0.5)) if std > 1e-8 else 0.0

        if sharpe < 0.3:
            prev_stage = promo.current_stage
            promo.current_stage = demotion_map[prev_stage]
            promo.rejection_reason = f"Auto-demoted: Sharpe {sharpe:.3f} < 0.3 over 90 days"
            promo.review_history = (promo.review_history or []) + [{
                "ts": datetime.now(UTC).isoformat(),
                "action": "auto_demoted",
                "from_stage": prev_stage,
                "to_stage": promo.current_stage,
                "sharpe": round(sharpe, 3),
                "num_trades": len(pnls),
            }]
            db.add(promo)
            demoted.append({"strategy": promo.strategy_name, "from": prev_stage, "to": promo.current_stage})

    await db.commit()
    return {"demoted": demoted, "count": len(demoted)}


@router.get("/daily-pnl/by-strategy")
async def get_daily_pnl_by_strategy(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Daily P&L breakdown per strategy — shows which desk is generating alpha."""
    account_ids = await _user_account_ids(db, current_user.id)
    since = datetime.now(UTC) - timedelta(days=days)

    if not account_ids:
        return {"strategies": [], "total_pnl": 0.0}

    result = await db.execute(
        select(
            Trade.strategy_name,
            func.date_trunc("day", Trade.closed_at).label("day"),
            func.sum(Trade.realized_pnl).label("daily_pnl"),
            func.count(Trade.id).label("n_trades"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= since,
            Trade.realized_pnl.isnot(None),
            Trade.strategy_name.isnot(None),
        )
        .group_by(Trade.strategy_name, func.date_trunc("day", Trade.closed_at))
        .order_by(Trade.strategy_name, func.date_trunc("day", Trade.closed_at))
    )
    rows = result.all()

    by_strategy: dict[str, dict] = {}
    for row in rows:
        name = row.strategy_name or "unknown"
        day_str = row.day.strftime("%Y-%m-%d") if hasattr(row.day, "strftime") else str(row.day)[:10]
        if name not in by_strategy:
            by_strategy[name] = {"series": [], "total_pnl": 0.0}
        pnl = float(row.daily_pnl or 0)
        by_strategy[name]["series"].append({"date": day_str, "pnl": round(pnl, 2), "trades": row.n_trades})
        by_strategy[name]["total_pnl"] = round(by_strategy[name]["total_pnl"] + pnl, 2)

    strategies = [
        {"strategy": name, **data}
        for name, data in sorted(by_strategy.items(), key=lambda x: -x[1]["total_pnl"])
    ]

    total_pnl = round(sum(s["total_pnl"] for s in strategies), 2)
    return {
        "strategies": strategies,
        "total_strategies": len(strategies),
        "total_pnl": total_pnl,
        "period_days": days,
    }


@router.get("/pipeline-status")
async def get_pipeline_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Full backtest → paper → shadow → staging → live pipeline status.
    Returns counts per stage and strategies ready for next promotion.
    """
    from app.models.promotion import StrategyPromotion
    from app.models.strategy import Strategy as StrategyModel

    promo_result = await db.execute(select(StrategyPromotion))
    promotions = promo_result.scalars().all()

    by_stage: dict[str, list[dict]] = {
        "paper": [], "shadow": [], "staging": [], "live": [], "rejected": []
    }
    ready_for_promotion = []

    for p in promotions:
        stage = p.current_stage if p.current_stage in by_stage else "paper"
        if stage == "live":
            stage_metrics = p.live_metrics or {}
        elif stage == "staging":
            stage_metrics = p.staging_metrics or {}
        elif stage == "shadow":
            stage_metrics = p.shadow_metrics or {}
        else:
            stage_metrics = p.paper_metrics or {}

        entry = {
            "id": p.id,
            "strategy_name": p.strategy_name,
            "sharpe": stage_metrics.get("sharpe", 0.0),
            "win_rate": stage_metrics.get("win_rate", 0.0),
            "max_drawdown": stage_metrics.get("max_drawdown", 0.0),
            "days_in_stage": stage_metrics.get("days_in_stage", 0),
            "promotion_ready": p.promotion_ready,
            "promotion_ready_stage": p.promotion_ready_stage,
        }
        by_stage[stage].append(entry)
        if p.promotion_ready and not p.awaiting_approval:
            ready_for_promotion.append({
                "id": p.id,
                "strategy": p.strategy_name,
                "current_stage": stage,
                "ready_for": p.promotion_ready_stage,
            })

    strat_result = await db.execute(
        select(func.count()).select_from(StrategyModel)
        .where(StrategyModel.is_active == True)
    )
    total_active = strat_result.scalar() or 0
    in_pipeline = sum(len(v) for v in by_stage.values())

    return {
        "pipeline": {stage: {"count": len(entries), "strategies": entries} for stage, entries in by_stage.items()},
        "ready_for_promotion": ready_for_promotion,
        "total_active_strategies": total_active,
        "total_in_pipeline": in_pipeline,
        "unregistered": max(0, total_active - in_pipeline),
        "computed_at": datetime.now(UTC).isoformat(),
    }


@router.get("/is-analysis")
async def get_is_analysis(
    days: int = Query(90, ge=30, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Implementation Shortfall (IS) analysis — institutional-grade execution quality.

    IS = (fill_price - arrival_price) / arrival_price * 10_000 bps
    Positive IS = you paid more than the mid at order time (implementation cost).
    Negative IS = you bought cheaper than the mid (good execution).

    Returns:
      overall_is_bps: float — average IS across all tracked orders
      vwap_shortfall_bps: float — how far fills deviated from period VWAP
      by_strategy: list — IS breakdown per strategy name (joined via Order → Trade)
      by_algo: list — IS by execution algorithm (twap/vwap/limit_first/market)
      by_size_bucket: list — IS bucketed by order size (small/medium/large)
      execution_efficiency_score: float — 0-100 score (100=perfect, negative IS = >100)
      trend_7d: float — IS trend over last 7d vs prior 7d (improvement if negative)
      data_available: bool
    """
    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return {
            "overall_is_bps": None,
            "vwap_shortfall_bps": None,
            "avg_execution_duration_seconds": None,
            "by_strategy": [],
            "by_algo": [],
            "by_size_bucket": [],
            "execution_efficiency_score": None,
            "trend_7d": None,
            "data_available": False,
        }

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Fetch SlippageRecords joined to Orders, filtered by user's accounts and date range
    stmt = (
        select(SlippageRecord)
        .join(Order, SlippageRecord.order_id == Order.id)
        .where(
            Order.account_id.in_(account_ids),
            SlippageRecord.created_at >= cutoff,
        )
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    if not records:
        return {
            "overall_is_bps": None,
            "vwap_shortfall_bps": None,
            "avg_execution_duration_seconds": None,
            "by_strategy": [],
            "by_algo": [],
            "by_size_bucket": [],
            "execution_efficiency_score": None,
            "trend_7d": None,
            "data_available": False,
        }

    # Helper: compute IS bps for a single record
    def _compute_is_bps(rec: SlippageRecord) -> float | None:
        if rec.is_cost_bps is not None:
            return float(rec.is_cost_bps)
        # Fallback: derive from fill_price and signal_price
        if rec.fill_price is not None and rec.signal_price is not None and float(rec.signal_price) != 0:
            return (float(rec.fill_price) - float(rec.signal_price)) / float(rec.signal_price) * 10_000
        return None

    # Helper: size bucket using signal_price as a proxy for notional
    def _size_bucket(rec: SlippageRecord) -> str:
        sp = float(rec.signal_price) if rec.signal_price is not None else 0.0
        if sp < 1_000:
            return "small"
        elif sp < 10_000:
            return "medium"
        else:
            return "large"

    # Collect all IS values
    is_values = [v for rec in records if (v := _compute_is_bps(rec)) is not None]
    vwap_values = [float(rec.vwap_shortfall_bps) for rec in records if rec.vwap_shortfall_bps is not None]
    duration_values = [float(rec.execution_duration_seconds) for rec in records if rec.execution_duration_seconds is not None]

    overall_is_bps = round(sum(is_values) / len(is_values), 4) if is_values else None
    avg_vwap_shortfall = round(sum(vwap_values) / len(vwap_values), 4) if vwap_values else None
    avg_duration = round(sum(duration_values) / len(duration_values), 2) if duration_values else None

    # by_algo breakdown
    algo_buckets: dict[str, list[float]] = {}
    for rec in records:
        algo = rec.execution_algo or "unknown"
        val = _compute_is_bps(rec)
        if val is not None:
            algo_buckets.setdefault(algo, []).append(val)

    by_algo = [
        {
            "algo": algo,
            "avg_is_bps": round(sum(vals) / len(vals), 4),
            "count": len(vals),
        }
        for algo, vals in sorted(algo_buckets.items())
    ]

    # by_strategy: join Order → strategy_id, then look up trade strategy_name
    # We fetch order_ids from our records and query trades matching account + symbol proximity
    order_ids = [rec.order_id for rec in records]
    order_stmt = select(Order).where(Order.id.in_(order_ids))
    order_result = await db.execute(order_stmt)
    orders_map = {o.id: o for o in order_result.scalars().all()}

    # Gather strategy names from trades that share account_id (best-effort join)
    strategy_buckets: dict[str, list[float]] = {}
    for rec in records:
        val = _compute_is_bps(rec)
        if val is None:
            continue
        order = orders_map.get(rec.order_id)
        strategy_name: str | None = None
        if order is not None and order.strategy_id is not None:
            # Try to find a matching trade with same account and strategy
            trade_stmt = (
                select(Trade.strategy_name)
                .where(
                    Trade.account_id == order.account_id,
                    Trade.strategy_id == order.strategy_id,
                )
                .limit(1)
            )
            trade_result = await db.execute(trade_stmt)
            strategy_name = trade_result.scalar()
        strategy_name = strategy_name or (order.strategy_id if order else None) or "unknown"
        strategy_buckets.setdefault(strategy_name, []).append(val)

    by_strategy = [
        {
            "strategy": strat,
            "avg_is_bps": round(sum(vals) / len(vals), 4),
            "count": len(vals),
        }
        for strat, vals in sorted(strategy_buckets.items())
    ]

    # by_size_bucket
    size_buckets: dict[str, list[float]] = {}
    for rec in records:
        val = _compute_is_bps(rec)
        if val is None:
            continue
        bucket = _size_bucket(rec)
        size_buckets.setdefault(bucket, []).append(val)

    by_size_bucket = [
        {
            "bucket": bucket,
            "avg_is_bps": round(sum(vals) / len(vals), 4),
            "count": len(vals),
        }
        for bucket in ("small", "medium", "large")
        if bucket in size_buckets
        for vals in [size_buckets[bucket]]
    ]

    # Execution efficiency score: 100 = zero IS, >100 if negative IS (price improvement)
    execution_efficiency_score: float | None = None
    if overall_is_bps is not None:
        raw_score = 100.0 - max(0.0, overall_is_bps)
        execution_efficiency_score = round(max(0.0, raw_score), 2)

    # Trend: compare avg IS in last 7d vs prior 7d
    trend_7d: float | None = None
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(days=7)
    prior_cutoff = now - timedelta(days=14)

    recent_vals = [
        v for rec in records
        if rec.created_at >= recent_cutoff
        if (v := _compute_is_bps(rec)) is not None
    ]
    prior_vals = [
        v for rec in records
        if prior_cutoff <= rec.created_at < recent_cutoff
        if (v := _compute_is_bps(rec)) is not None
    ]

    if recent_vals and prior_vals:
        recent_avg = sum(recent_vals) / len(recent_vals)
        prior_avg = sum(prior_vals) / len(prior_vals)
        trend_7d = round(recent_avg - prior_avg, 4)  # negative = improvement

    return {
        "overall_is_bps": overall_is_bps,
        "vwap_shortfall_bps": avg_vwap_shortfall,
        "avg_execution_duration_seconds": avg_duration,
        "by_strategy": by_strategy,
        "by_algo": by_algo,
        "by_size_bucket": by_size_bucket,
        "execution_efficiency_score": execution_efficiency_score,
        "trend_7d": trend_7d,
        "data_available": True,
        "record_count": len(records),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/factor-attribution")
async def get_factor_attribution(
    days: int = Query(90, ge=30, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fama-French 5-factor attribution of realized P&L.

    Uses free yfinance data for factor proxies:
      - Market (MKT-RF): SPY daily returns - 0.04/252 (risk-free rate proxy)
      - Size (SMB): IWM - SPY (small minus big)
      - Value (HML): IVE - IVW (iShares S&P 500 Value vs Growth)
      - Momentum (MOM): MTUM - VTV (momentum ETF vs value)
      - Quality (QMJ): QUAL - USMV (quality vs min vol)

    Regresses daily strategy P&L (from Trade records) against these 5 factors
    using OLS (numpy lstsq). Returns factor loadings (betas), R-squared,
    alpha (Jensen's alpha annualized), and % of variance explained per factor.

    Returns:
      alpha_annualized_pct: float — Jensen's alpha annualized
      r_squared: float — goodness of fit (0-1)
      factors: dict — {factor_name: {beta, t_stat, contribution_pct}}
      total_explained_pct: float — % of P&L variance explained by the 5 factors
      unexplained_pct: float — alpha + noise
      data_start: str ISO date
      data_end: str ISO date
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not available"}

    import numpy as np

    account_ids = await _user_account_ids(db, current_user.id)
    if not account_ids:
        return {"error": "insufficient_data", "min_days_needed": 20}

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Fetch daily realized P&L grouped by date(closed_at)
    pnl_stmt = (
        select(
            func.date(Trade.closed_at).label("trade_date"),
            func.sum(Trade.realized_pnl).label("daily_pnl"),
        )
        .where(
            Trade.account_id.in_(account_ids),
            Trade.closed_at >= cutoff,
        )
        .group_by(func.date(Trade.closed_at))
        .order_by(func.date(Trade.closed_at))
    )
    pnl_result = await db.execute(pnl_stmt)
    pnl_rows = pnl_result.all()

    if not pnl_rows or len(pnl_rows) < 20:
        return {"error": "insufficient_data", "min_days_needed": 20}

    # Build daily P&L series indexed by date
    pnl_by_date: dict[date, float] = {}
    for row in pnl_rows:
        trade_date = row.trade_date
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
        elif isinstance(trade_date, datetime):
            trade_date = trade_date.date()
        pnl_by_date[trade_date] = float(row.daily_pnl)

    # Download factor ETFs via yfinance
    etf_tickers = ["SPY", "IWM", "IVE", "IVW", "MTUM", "VTV", "QUAL", "USMV"]
    try:
        raw = yf.download(etf_tickers, period="1y", auto_adjust=True, progress=False)["Close"]
        if hasattr(raw, "columns"):
            # Multi-ticker download returns DataFrame with ticker columns
            prices = raw
        else:
            return {"error": "yfinance returned unexpected data format"}
    except Exception as exc:
        logger.warning("yfinance download failed: %s", exc)
        return {"error": "factor_data_unavailable", "detail": str(exc)}

    # Compute daily factor returns
    prices = prices.dropna(how="all")
    if prices.empty or len(prices) < 5:
        return {"error": "insufficient_factor_data"}

    # Ensure all required tickers are present
    missing = [t for t in etf_tickers if t not in prices.columns]
    if missing:
        return {"error": "missing_factor_tickers", "missing": missing}

    pct = prices.pct_change().dropna()

    risk_free_daily = 0.04 / 252

    factor_series: dict[str, pd.Series] = {
        "MKT-RF": pct["SPY"] - risk_free_daily,
        "SMB":    pct["IWM"] - pct["SPY"],
        "HML":    pct["IVE"] - pct["IVW"],
        "MOM":    pct["MTUM"] - pct["VTV"],
        "QMJ":    pct["QUAL"] - pct["USMV"],
    }

    # Convert to DataFrame for easy alignment
    factor_df = pd.DataFrame(factor_series)
    factor_df.index = pd.to_datetime(factor_df.index).date  # type: ignore[assignment]

    # Build strategy P&L series aligned to factor dates (inner join)
    common_dates = sorted(set(factor_df.index) & set(pnl_by_date.keys()))

    if len(common_dates) < 20:
        return {"error": "insufficient_data", "min_days_needed": 20}

    y = np.array([pnl_by_date[d] for d in common_dates], dtype=float)
    X_factors = factor_df.loc[common_dates].values.astype(float)

    # Add constant (intercept) for alpha
    ones = np.ones((len(y), 1), dtype=float)
    X = np.hstack([ones, X_factors])  # shape (n, 6)

    # OLS: solve X @ beta = y
    coeffs, residuals_arr, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    alpha_daily = float(coeffs[0])
    betas = coeffs[1:]  # shape (5,)

    # Compute R-squared
    y_pred = X @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # Standard errors for t-stats
    n = len(y)
    k = X.shape[1]  # 6 (intercept + 5 factors)
    residuals = y - y_pred
    if n > k:
        sigma2 = float(np.sum(residuals ** 2)) / (n - k)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(sigma2 * np.diag(XtX_inv), 0.0))
        t_stats = coeffs / (se + 1e-12)
    else:
        se = np.zeros(k)
        t_stats = np.zeros(k)

    # Factor contribution: beta_i * std(factor_i) / std(y_pred) * 100
    factor_std = X_factors.std(axis=0)
    y_pred_std = float(y_pred.std()) if y_pred.std() > 1e-12 else 1.0
    contribution_pcts = [
        float(abs(betas[i]) * factor_std[i] / y_pred_std * 100)
        for i in range(5)
    ]
    total_explained = round(min(100.0, sum(contribution_pcts)), 2)

    # Annualize alpha: daily alpha * 252 * 100 (to percent)
    alpha_annualized_pct = round(alpha_daily * 252 * 100, 4)

    factor_names = list(factor_series.keys())
    factors_out = {
        factor_names[i]: {
            "beta": round(float(betas[i]), 6),
            "t_stat": round(float(t_stats[i + 1]), 4),  # +1 to skip intercept t_stat
            "contribution_pct": round(contribution_pcts[i], 2),
        }
        for i in range(5)
    }

    return {
        "alpha_annualized_pct": alpha_annualized_pct,
        "r_squared": round(r_squared, 6),
        "factors": factors_out,
        "total_explained_pct": total_explained,
        "unexplained_pct": round(max(0.0, 100.0 - total_explained), 2),
        "data_start": str(common_dates[0]),
        "data_end": str(common_dates[-1]),
        "n_observations": len(common_dates),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Funding Rate Monitor ─────────────────────────────────────────────────────

# Binance perpetual futures funding rates (public API, no auth required)
_BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_BINANCE_PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"

# Tracked perp pairs — top institutional crypto
_FUNDING_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "MATICUSDT", "DOTUSDT",
    "LINKUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
]


async def _fetch_binance_funding(symbol: str, limit: int = 8) -> list[dict]:
    """Fetch the last N funding rate events for a Binance perp symbol."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                _BINANCE_FUNDING_URL,
                params={"symbol": symbol, "limit": limit},
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as exc:
        logger.debug("Binance funding fetch failed", symbol=symbol, error=str(exc))
    return []


async def _fetch_binance_premium_index() -> list[dict]:
    """Fetch current mark price + predicted funding rate for all perps."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_BINANCE_PREMIUM_INDEX_URL)
            if resp.status_code == 200:
                return resp.json()
    except Exception as exc:
        logger.debug("Binance premium index fetch failed", error=str(exc))
    return []


@router.get("/funding-rates")
async def get_funding_rates(
    symbols: str | None = Query(None, description="Comma-separated perp symbols, e.g. BTCUSDT,ETHUSDT"),
    current_user: User = Depends(get_current_user),
):
    """
    Crypto perpetual futures funding rate monitor.

    Fetches live funding rates from Binance public API (no auth required).
    Returns current rate, predicted next rate, 24h average, annualized rate,
    and the last 8 funding events per symbol.

    Funding rate > 0 means longs pay shorts (bullish sentiment, arb: short perp + long spot).
    Funding rate < 0 means shorts pay longs (bearish sentiment, arb: long perp + short spot).

    Returns:
      symbols: list of {
        symbol, base_asset, mark_price, index_price,
        last_funding_rate, next_funding_time,
        rate_annualized_pct, avg_rate_8h, signal,
        history: [{fundingTime, fundingRate}]
      }
      computed_at: ISO timestamp
      arb_opportunities: list of symbols where |rate| > 0.1% (10bps) per 8h
    """
    target_symbols = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols
        else _FUNDING_SYMBOLS
    )

    # Fetch premium index for all symbols at once (one call)
    premium_index = await _fetch_binance_premium_index()
    premium_map: dict[str, dict] = {
        row["symbol"]: row for row in premium_index
        if row.get("symbol") in target_symbols
    }

    # Fetch history for each tracked symbol concurrently
    import asyncio
    history_tasks = [_fetch_binance_funding(sym, limit=8) for sym in target_symbols]
    histories = await asyncio.gather(*history_tasks)
    history_map: dict[str, list[dict]] = dict(zip(target_symbols, histories))

    out = []
    arb_opportunities = []

    for sym in target_symbols:
        pm = premium_map.get(sym, {})
        history = history_map.get(sym, [])

        # Parse latest funding rate from history (most recent entry)
        last_rate: float | None = None
        if history:
            try:
                last_rate = float(history[-1]["fundingRate"])
            except (KeyError, ValueError, TypeError):
                last_rate = None

        # Parse current predicted rate from premium index
        predicted_rate: float | None = None
        try:
            predicted_rate = float(pm.get("lastFundingRate") or 0.0)
        except (ValueError, TypeError):
            predicted_rate = None

        # Mark / index prices
        mark_price: float | None = None
        index_price: float | None = None
        try:
            mark_price = float(pm.get("markPrice") or 0.0) or None
            index_price = float(pm.get("indexPrice") or 0.0) or None
        except (ValueError, TypeError):
            pass

        # Next funding time (epoch ms)
        next_funding_time: str | None = None
        try:
            nft = pm.get("nextFundingTime")
            if nft:
                next_funding_time = datetime.fromtimestamp(int(nft) / 1000, tz=UTC).isoformat()
        except Exception:
            pass

        # 8-period average rate
        rates = []
        for h in history:
            try:
                rates.append(float(h["fundingRate"]))
            except (KeyError, ValueError, TypeError):
                pass
        avg_rate_8h = round(sum(rates) / len(rates), 8) if rates else None

        # Annualized: funding paid 3x per day → 3 * 365 = 1095 periods per year
        rate_annualized_pct: float | None = None
        if avg_rate_8h is not None:
            rate_annualized_pct = round(avg_rate_8h * 1095 * 100, 4)

        # Trading signal
        signal = "neutral"
        if last_rate is not None:
            abs_rate = abs(last_rate)
            if abs_rate > 0.001:  # > 10 bps per 8h = extreme
                signal = "sell_perp_buy_spot" if last_rate > 0 else "buy_perp_sell_spot"
            elif abs_rate > 0.0003:  # > 3 bps
                signal = "slight_long_bias" if last_rate > 0 else "slight_short_bias"

        base_asset = sym.replace("USDT", "").replace("BUSD", "")

        entry = {
            "symbol": sym,
            "base_asset": base_asset,
            "mark_price": round(mark_price, 6) if mark_price else None,
            "index_price": round(index_price, 6) if index_price else None,
            "last_funding_rate": round(last_rate, 8) if last_rate is not None else None,
            "last_funding_rate_pct": round(last_rate * 100, 6) if last_rate is not None else None,
            "predicted_rate": round(predicted_rate, 8) if predicted_rate is not None else None,
            "next_funding_time": next_funding_time,
            "avg_rate_8h": avg_rate_8h,
            "rate_annualized_pct": rate_annualized_pct,
            "signal": signal,
            "history": [
                {
                    "funding_time": datetime.fromtimestamp(
                        int(h["fundingTime"]) / 1000, tz=datetime.now(UTC).tzinfo
                    ).isoformat() if h.get("fundingTime") else None,
                    "rate": float(h["fundingRate"]) if h.get("fundingRate") else None,
                    "rate_pct": round(float(h["fundingRate"]) * 100, 6) if h.get("fundingRate") else None,
                }
                for h in history
            ],
        }
        out.append(entry)

        # Flag as arb opportunity if |rate| > 10bps per 8h
        if last_rate is not None and abs(last_rate) >= 0.001:
            arb_opportunities.append({
                "symbol": sym,
                "base_asset": base_asset,
                "rate_pct": round(last_rate * 100, 4),
                "annualized_pct": rate_annualized_pct,
                "direction": "short_perp" if last_rate > 0 else "long_perp",
                "signal": signal,
            })

    # Sort by abs rate descending
    out.sort(key=lambda x: abs(x.get("last_funding_rate") or 0), reverse=True)
    arb_opportunities.sort(key=lambda x: abs(x["rate_pct"]), reverse=True)

    return {
        "symbols": out,
        "arb_opportunities": arb_opportunities,
        "total_symbols": len(out),
        "extreme_count": sum(1 for x in out if abs(x.get("last_funding_rate") or 0) >= 0.001),
        "computed_at": datetime.now(UTC).isoformat(),
    }
