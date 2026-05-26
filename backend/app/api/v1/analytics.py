"""Analytics and performance metrics endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
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
