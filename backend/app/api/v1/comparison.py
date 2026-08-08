"""API endpoints for comparing manual and ML strategies.

Provides routes to fetch benchmark statistics and recent comparison results.
"""
from typing import Any, List, Optional

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
WINNER_MANUAL = "manual"
WINNER_ML = "ml"

router = APIRouter(prefix=PREFIX, tags=[TAG])


class ComparisonOut(BaseModel):
    """Schema representing a comparison between manual and ML strategies.

    Attributes
    ----------
    id : str
        Unique identifier of the comparison record.
    strategy_name : str
        Name of the strategy being compared.
    symbol : str
        Trading symbol (e.g., ticker) associated with the comparison.
    manual_sharpe : float | None
        Sharpe ratio of the manual strategy, if available.
    ml_sharpe : float | None
        Sharpe ratio of the ML-driven strategy, if available.
    is_significant : bool | None
        Indicator whether the performance difference is statistically significant.
    winner : str | None
        Identifier of the winning side ('manual' or 'ml'), if determined.
    spy_sharpe : float | None
        Benchmark Sharpe ratio (e.g., SPY) for reference.
    ml_improvement_pct : float | None
        Percentage improvement of the ML Sharpe over the manual Sharpe,
        rounded to ``IMPROVEMENT_PRECISION`` decimal places.
    """

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
    def from_model(cls, m: ComparisonModel) -> "ComparisonOut":
        """Create a ``ComparisonOut`` instance from a database model.

        Parameters
        ----------
        m : ComparisonModel
            ORM model instance containing raw comparison data.

        Returns
        -------
        ComparisonOut
            Pydantic model populated with transformed and rounded values.
        """
        improvement: Optional[float] = None
        if m.manual_sharpe is not None and m.ml_sharpe is not None:
            # Use a minimal base to avoid division by zero
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
async def get_benchmarks() -> Any:
    """Retrieve benchmark statistics for strategy comparison.

    Returns
    -------
    Any
        The raw benchmark data returned by ``get_benchmark_stats``.
    """
    return get_benchmark_stats()


@router.get(ENDPOINT_RESULTS, response_model=list[ComparisonOut])
@router.get(ENDPOINT_ROOT, response_model=list[ComparisonOut])
async def list_comparisons(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[ComparisonOut]:
    """List recent comparison results for the current user.

    Queries the database for the most recent ``ComparisonModel`` entries,
    limited by ``DEFAULT_LIMIT``, and returns them as ``ComparisonOut`` objects.

    Parameters
    ----------
    db : AsyncSession
        Asynchronous database session provided by FastAPI dependency injection.
    current_user : User
        Authenticated user obtained via ``get_current_user`` dependency.

    Returns
    -------
    List[ComparisonOut]
        A list of transformed comparison results.
    """
    # Guard against unexpected None from the DB layer
    result = await db.execute(
        select(ComparisonModel)
        .order_by(ComparisonModel.created_at.desc())
        .limit(max(DEFAULT_LIMIT, 1))
    )
    rows: List[ComparisonModel] = result.scalars().all() if result is not None else []

    # Ensure we always return a list, even if no rows are found
    if not rows:
        return []

    return [ComparisonOut.from_model(r) for r in rows]