"""Manual vs ML strategy comparison endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.comparison import ComparisonResult as ComparisonModel
from app.models.user import User
from app.comparison.benchmarks import get_benchmark_stats
from pydantic import BaseModel, ConfigDict, ConfigDict
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
    def from_model(cls, m) -> "ComparisonOut":
        improvement = None
        if m.manual_sharpe is not None and m.ml_sharpe is not None:
            base = float(m.manual_sharpe) or 1e-9
            improvement = (float(m.ml_sharpe) - float(m.manual_sharpe)) / abs(base)
        return cls(
            id=m.id, strategy_name=m.strategy_name, symbol=m.symbol,
            manual_sharpe=float(m.manual_sharpe) if m.manual_sharpe else None,
            ml_sharpe=float(m.ml_sharpe) if m.ml_sharpe else None,
            is_significant=m.is_significant, winner=m.winner,
            spy_sharpe=float(m.spy_sharpe) if m.spy_sharpe else None,
            ml_improvement_pct=round(improvement, 4) if improvement else None,
        )


@router.get("/benchmarks")
async def get_benchmarks():
    return get_benchmark_stats()


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
