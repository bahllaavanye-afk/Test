"""Manual vs ML strategy comparison endpoints."""
from datetime import date
from typing import Optional, List

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.comparison.benchmarks import get_benchmark_stats
from app.database import get_db
from app.models.comparison import ComparisonResult as ComparisonModel
from app.models.user import User
from pydantic import BaseModel, ConfigDict, Field, validator


router = APIRouter(prefix="/comparison", tags=["comparison"])


class ComparisonOut(BaseModel):
    """Schema representing a single comparison between manual and ML strategies."""

    id: str = Field(..., description="Unique identifier of the comparison record.", example="c3f9b2e1-4d5a-4f2a-9a1e-2b6c7d8e9f0a")
    strategy_name: str = Field(..., description="Human‑readable name of the strategy under test.", example="mean_rev_20_2")
    symbol: str = Field(..., description="Ticker symbol the strategy was applied to.", example="SPY")
    manual_sharpe: Optional[float] = Field(
        None,
        description="Sharpe ratio of the manual (baseline) strategy.",
        example=0.85,
        ge=0,
    )
    ml_sharpe: Optional[float] = Field(
        None,
        description="Sharpe ratio of the ML‑enhanced strategy.",
        example=1.12,
        ge=0,
    )
    is_significant: Optional[bool] = Field(
        None,
        description="Whether the performance difference is statistically significant.",
        example=True,
    )
    winner: Optional[str] = Field(
        None,
        description="Identifier of the winning approach: 'manual', 'ml', or 'draw'.",
        example="ml",
    )
    spy_sharpe: Optional[float] = Field(
        None,
        description="Sharpe ratio of the SPY benchmark for the same period.",
        example=0.95,
        ge=0,
    )
    ml_improvement_pct: Optional[float] = Field(
        None,
        description="Percentage improvement of the ML Sharpe over the manual Sharpe, rounded to 4 decimals.",
        example=0.3176,
        ge=-1,
        le=10,
    )

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_model(cls, m: ComparisonModel) -> "ComparisonOut":
        """Factory method to create a schema instance from a SQLAlchemy model."""
        improvement = None
        if m.manual_sharpe is not None and m.ml_sharpe is not None:
            base = float(m.manual_sharpe) or 1e-9
            improvement = (float(m.ml_sharpe) - float(m.manual_sharpe)) / abs(base)
        return cls(
            id=str(m.id),
            strategy_name=m.strategy_name,
            symbol=m.symbol,
            manual_sharpe=float(m.manual_sharpe) if m.manual_sharpe is not None else None,
            ml_sharpe=float(m.ml_sharpe) if m.ml_sharpe is not None else None,
            is_significant=m.is_significant,
            winner=m.winner,
            spy_sharpe=float(m.spy_sharpe) if m.spy_sharpe is not None else None,
            ml_improvement_pct=round(improvement, 4) if improvement is not None else None,
        )

    @validator("ml_improvement_pct", always=True)
    def validate_improvement(cls, v, values):
        """Ensures the improvement percentage is rounded and within a reasonable range."""
        if v is None:
            return None
        return round(v, 4)


@router.get("/benchmarks")
async def get_benchmarks():
    """Return benchmark statistics for comparison."""
    return get_benchmark_stats()


@router.get("/results", response_model=List[ComparisonOut])
@router.get("/", response_model=List[ComparisonOut])
async def list_comparisons(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List recent comparison results for the authenticated user."""
    result = await db.execute(
        select(ComparisonModel).order_by(ComparisonModel.created_at.desc()).limit(20)
    )
    rows = result.scalars().all()
    return [ComparisonOut.from_model(r) for r in rows]