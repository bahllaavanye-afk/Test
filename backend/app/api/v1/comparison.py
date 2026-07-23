"""API endpoints for comparing manual and ML trading strategies."""

from datetime import date
from typing import Any, List, Optional

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


class ComparisonOut(BaseModel):
    """Response model representing a comparison between manual and ML strategies."""

    id: str
    strategy_name: str
    symbol: str
    manual_sharpe: Optional[float] = None
    ml_sharpe: Optional[float] = None
    is_significant: Optional[bool] = None
    winner: Optional[str] = None
    spy_sharpe: Optional[float] = None
    ml_improvement_pct: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_model(cls, m: ComparisonModel) -> "ComparisonOut":
        """
        Create a ``ComparisonOut`` instance from a SQLAlchemy model.

        Args:
            m: The ``ComparisonModel`` instance fetched from the database.

        Returns:
            An instance of ``ComparisonOut`` populated with values from ``m``.
        """
        improvement: Optional[float] = None
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


@router.get("/benchmarks", summary="Retrieve benchmark statistics")
async def get_benchmarks() -> dict[str, Any]:
    """
    Return benchmark statistics used for strategy comparison.

    The function delegates to :func:`get_benchmark_stats` which aggregates
    reference metrics such as SPY performance.

    Returns:
        A dictionary containing benchmark data.
    """
    return get_benchmark_stats()


@router.get(
    "/results",
    response_model=List[ComparisonOut],
    summary="List recent strategy comparisons (alias)",
)
@router.get(
    "/",
    response_model=List[ComparisonOut],
    summary="List recent strategy comparisons",
)
async def list_comparisons(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[ComparisonOut]:
    """
    Retrieve the most recent strategy comparison records for the authenticated user.

    Args:
        db: The asynchronous SQLAlchemy session provided by the dependency injector.
        current_user: The currently authenticated user.

    Returns:
        A list of ``ComparisonOut`` objects representing the latest comparisons,
        ordered by creation time descending and limited to the 20 most recent.
    """
    result = await db.execute(
        select(ComparisonModel).order_by(ComparisonModel.created_at.desc()).limit(20)
    )
    rows = result.scalars().all()
    return [ComparisonOut.from_model(r) for r in rows]