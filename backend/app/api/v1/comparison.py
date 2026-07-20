"""Manual vs ML strategy comparison endpoints."""
import logging
import time
from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.comparison.benchmarks import get_benchmark_stats
from app.database import get_db
from app.models.comparison import ComparisonResult as ComparisonModel
from app.models.user import User
from pydantic import BaseModel, ConfigDict

router = APIRouter(prefix="/comparison", tags=["comparison"])

logger = logging.getLogger(__name__)


class ComparisonOut(BaseModel):
    id: str
    strategy_name: str
    symbol: str
    manual_sharpe: float | None
    ml_sharpe: float | None
    is_significant: bool | None
    winner: str | None
    spy_sharpe: float | None
    ml_improvement_pct: float | None = None

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_model(cls, m) -> "ComparisonOut":
        improvement = None
        if m.manual_sharpe is not None and m.ml_sharpe is not None:
            base = float(m.manual_sharpe) or 1e-9
            improvement = (float(m.ml_sharpe) - float(m.manual_sharpe)) / abs(base)
        return cls(
            id=m.id,
            strategy_name=m.strategy_name,
            symbol=m.symbol,
            manual_sharpe=float(m.manual_sharpe) if m.manual_sharpe else None,
            ml_sharpe=float(m.ml_sharpe) if m.ml_sharpe else None,
            is_significant=m.is_significant,
            winner=m.winner,
            spy_sharpe=float(m.spy_sharpe) if m.spy_sharpe else None,
            ml_improvement_pct=round(improvement, 4) if improvement else None,
        )


@router.get("/benchmarks")
async def get_benchmarks():
    return get_benchmark_stats()


@router.get("/results", response_model=list[ComparisonOut])
@router.get("/", response_model=list[ComparisonOut])
async def list_comparisons(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start_time = time.perf_counter()
    result = await db.execute(
        select(ComparisonModel).order_by(ComparisonModel.created_at.desc()).limit(20)
    )
    rows = result.scalars().all()
    comparisons = [ComparisonOut.from_model(r) for r in rows]

    # Metrics for structured logging
    signal_count = len(comparisons)
    execution_time_ms = (time.perf_counter() - start_time) * 1000
    # Compute aggregate P&L as average ml_improvement_pct if available
    improvements = [
        comp.ml_improvement_pct for comp in comparisons if comp.ml_improvement_pct is not None
    ]
    avg_improvement = (
        sum(improvements) / len(improvements) if improvements else None
    )

    logger.info(
        "Fetched comparison results",
        extra={
            "signal_count": signal_count,
            "execution_time_ms": round(execution_time_ms, 2),
            "avg_ml_improvement_pct": avg_improvement,
            "user_id": getattr(current_user, "id", None),
        },
    )
    return comparisons