"""Analytics and performance metrics endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from app.database import get_db
from app.api.deps import get_current_user
from app.models.trade import Trade
from app.models.slippage import SlippageRecord
from app.models.user import User

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
