"""Manual vs ML strategy comparison endpoints."""
from datetime import date

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.comparison.benchmarks import get_benchmark_stats
from app.database import get_db
from app.models.comparison import ComparisonResult as ComparisonModel
from app.models.user import User

router = APIRouter(prefix="/comparison", tags=["comparison"])


def _to_float_or_none(value) -> float | None:
    """Convert a possibly‑None numeric value to float or None."""
    return float(value) if value is not None else None


def _calculate_improvement(manual_sharpe, ml_sharpe) -> float | None:
    """Calculate percentage improvement of ML Sharpe over manual Sharpe."""
    if manual_sharpe is None or ml_sharpe is None:
        return None
    base = float(manual_sharpe) or 1e-9
    improvement = (float(ml_sharpe) - float(manual_sharpe)) / abs(base)
    return round(improvement, 4)


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
        return cls(
            id=m.id,
            strategy_name=m.strategy_name,
            symbol=m.symbol,
            manual_sharpe=_to_float_or_none(m.manual_sharpe),
            ml_sharpe=_to_float_or_none(m.ml_sharpe),
            is_significant=m.is_significant,
            winner=m.winner,
            spy_sharpe=_to_float_or_none(m.spy_sharpe),
            ml_improvement_pct=_calculate_improvement(m.manual_sharpe, m.ml_sharpe),
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
    result = await db.execute(
        select(ComparisonModel).order_by(ComparisonModel.created_at.desc()).limit(20)
    )
    rows = result.scalars().all()
    return [ComparisonOut.from_model(r) for r in rows]