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
    """API response model for a single comparison result."""

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
    def from_model(cls, m: ComparisonModel | None) -> "ComparisonOut | None":
        """Create a ComparisonOut from a SQLAlchemy model instance.

        Handles ``None`` model instances gracefully.
        """
        if m is None:
            return None

        improvement = None
        if m.manual_sharpe is not None and m.ml_sharpe is not None:
            # Avoid division by zero; use a tiny epsilon if manual_sharpe is zero.
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
    """Return benchmark statistics.

    Guarantees a dictionary is returned; if the underlying function returns ``None``,
    an empty dict is provided to avoid runtime errors.
    """
    stats = get_benchmark_stats()
    return stats if stats is not None else {}


@router.get("/results", response_model=list[ComparisonOut])
@router.get("/", response_model=list[ComparisonOut])
async def list_comparisons(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List the most recent comparison results.

    Returns up to 20 recent entries, handling empty result sets safely.
    """
    # Defensive: ensure ``db`` and ``current_user`` are present.
    if db is None or current_user is None:
        return []

    result = await db.execute(
        select(ComparisonModel).order_by(ComparisonModel.created_at.desc()).limit(20)
    )
    rows = result.scalars().all() if result is not None else []

    # Guard against ``None`` rows and filter out any ``None`` entries.
    if not rows:
        return []

    return [item for r in rows if (item := ComparisonOut.from_model(r)) is not None]