"""Manual vs ML strategy comparison endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.comparison import ComparisonResult as ComparisonModel
from app.models.user import User
from app.comparison.benchmarks import get_benchmark_stats
from pydantic import BaseModel, ConfigDict
from datetime import date

router = APIRouter(prefix="/comparison", tags=["comparison"])


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
    def from_model(cls, m) -> "ComparisonOut | None":
        if m is None:
            return None
        improvement = None
        if m.manual_sharpe is not None and m.ml_sharpe is not None:
            base = float(m.manual_sharpe) or 1e-9
            improvement = (float(m.ml_sharpe) - float(m.manual_sharpe)) / abs(base)
        return cls(
            id=m.id,
            strategy_name=m.strategy_name,
            symbol=m.symbol,
            manual_sharpe=float(m.manual_sharpe) if m.manual_sharpe is not None else None,
            ml_sharpe=float(m.ml_sharpe) if m.ml_sharpe is not None else None,
            is_significant=m.is_significant,
            winner=m.winner,
            spy_sharpe=float(m.spy_sharpe) if m.spy_sharpe is not None else None,
            ml_improvement_pct=round(improvement, 4) if improvement is not None else None,
        )


@router.get("/benchmarks")
async def get_benchmarks():
    """Return benchmark statistics, ensuring a safe response for edge cases."""
    try:
        stats = get_benchmark_stats()
        # Defensive: if None or not a dict, return empty dict
        if not isinstance(stats, dict):
            return {}
        return stats
    except Exception:
        # In production we would log the exception; return empty dict to avoid breaking callers
        return {}


@router.get("/results", response_model=list[ComparisonOut])
@router.get("/", response_model=list[ComparisonOut])
async def list_comparisons(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List recent comparison results with safe handling of empty or missing data."""
    try:
        result = await db.execute(
            select(ComparisonModel).order_by(ComparisonModel.created_at.desc()).limit(20)
        )
    except Exception:
        # If the query fails, return an empty list rather than propagating an error
        return []

    if result is None:
        return []

    rows = result.scalars().all() or []
    # Filter out any None entries that might appear unexpectedly
    comparisons = [ComparisonOut.from_model(r) for r in rows if r is not None]
    # Ensure the response is always a list (even if empty)
    return comparisons if comparisons is not None else []