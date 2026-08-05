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
        """Create a ComparisonOut from a ComparisonModel handling edge cases."""
        if m is None:
            raise ValueError("Comparison model instance cannot be None")

        improvement = None
        # Ensure both sharpe values are present before computing improvement
        if m.manual_sharpe is not None and m.ml_sharpe is not None:
            # Guard against zero or falsy manual_sharpe to avoid division by zero
            base = float(m.manual_sharpe) if float(m.manual_sharpe) != 0 else MIN_MANUAL_SHARPE
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
            ml_improvement_pct=round(improvement, IMPROVEMENT_PRECISION) if improvement is not None else None,
        )


@router.get(ENDPOINT_BENCHMARKS)
async def get_benchmarks():
    """Return benchmark statistics, defaulting to an empty dict if None."""
    stats = get_benchmark_stats()
    return stats if stats is not None else {}


@router.get(ENDPOINT_RESULTS, response_model=list[ComparisonOut])
@router.get(ENDPOINT_ROOT, response_model=list[ComparisonOut])
async def list_comparisons(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List recent comparison results, safely handling empty query results."""
    # Ensure a sensible limit; fallback to DEFAULT_LIMIT if invalid
    limit = DEFAULT_LIMIT if isinstance(DEFAULT_LIMIT, int) and DEFAULT_LIMIT > 0 else 20

    result = await db.execute(
        select(ComparisonModel).order_by(ComparisonModel.created_at.desc()).limit(limit)
    )
    rows = result.scalars().all() or []
    return [ComparisonOut.from_model(r) for r in rows]