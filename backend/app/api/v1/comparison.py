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

# Constants
PREFIX = "/comparison"
TAG = "comparison"
ENDPOINT_BENCHMARKS = "/benchmarks"
ENDPOINT_RESULTS = "/results"
ENDPOINT_ROOT = "/"
MIN_MANUAL_SHARPE = 1e-9
IMPROVEMENT_PRECISION = 4
DEFAULT_LIMIT = 20

router = APIRouter(prefix=PREFIX, tags=[TAG])


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
        if not hasattr(m, "id"):
            raise ValueError("Comparison model must have an 'id' attribute")
        improvement = None
        if m.manual_sharpe is not None and m.ml_sharpe is not None:
            base = float(m.manual_sharpe) or MIN_MANUAL_SHARPE
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
            ml_improvement_pct=round(improvement, IMPROVEMENT_PRECISION) if improvement else None,
        )


@router.get(ENDPOINT_BENCHMARKS)
async def get_benchmarks():
    """Return benchmark statistics."""
    return get_benchmark_stats()


@router.get(ENDPOINT_RESULTS, response_model=list[ComparisonOut])
@router.get(ENDPOINT_ROOT, response_model=list[ComparisonOut])
async def list_comparisons(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List recent comparison results for the current user."""
    if not isinstance(db, AsyncSession):
        raise ValueError("Invalid database session provided")
    if not isinstance(current_user, User):
        raise ValueError("Invalid user context provided")
    result = await db.execute(
        select(ComparisonModel).order_by(ComparisonModel.created_at.desc()).limit(DEFAULT_LIMIT)
    )
    rows = result.scalars().all()
    return [ComparisonOut.from_model(r) for r in rows]