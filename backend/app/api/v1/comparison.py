"""Manual vs ML strategy comparison endpoints."""
from datetime import date
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.comparison.benchmarks import get_benchmark_stats
from app.database import get_db
from app.models.comparison import ComparisonResult as ComparisonModel
from app.models.user import User
from pydantic import BaseModel, ConfigDict, Field, field_validator


router = APIRouter(prefix="/comparison", tags=["comparison"])


class ComparisonOut(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier of the comparison record",
        example="c3f9a8e2-1b5d-4f2a-9c6e-3a8b4d5f6a7b",
    )
    strategy_name: str = Field(
        ...,
        description="Name of the strategy being compared",
        example="mean_rev_20_2",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol of the asset",
        example="SPY",
    )
    manual_sharpe: float | None = Field(
        None,
        description="Sharpe ratio of the manual strategy",
        example=1.23,
    )
    ml_sharpe: float | None = Field(
        None,
        description="Sharpe ratio of the ML strategy",
        example=1.45,
    )
    is_significant: bool | None = Field(
        None,
        description="Whether the performance difference is statistically significant",
        example=True,
    )
    winner: str | None = Field(
        None,
        description="Identifier of the winning strategy ('manual' or 'ml')",
        example="ml",
    )
    spy_sharpe: float | None = Field(
        None,
        description="Sharpe ratio of the SPY benchmark",
        example=1.1,
    )
    ml_improvement_pct: float | None = Field(
        None,
        description="Percentage improvement of ML over manual strategy",
        example=0.1795,
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("id")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        """Ensure the id is a valid UUID string."""
        import uuid

        try:
            uuid.UUID(v)
        except ValueError as exc:
            raise ValueError("id must be a valid UUID") from exc
        return v

    @field_validator("winner")
    @classmethod
    def validate_winner(cls, v: str | None) -> str | None:
        """Validate allowed values for winner."""
        if v is not None and v not in {"manual", "ml"}:
            raise ValueError("winner must be 'manual', 'ml', or None")
        return v

    @field_validator("ml_improvement_pct")
    @classmethod
    def round_improvement(cls, v: float | None) -> float | None:
        """Round improvement percentage to 4 decimal places."""
        if v is not None:
            return round(v, 4)
        return v

    @classmethod
    def from_model(cls, m: ComparisonModel) -> "ComparisonOut":
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
            ml_improvement_pct=improvement,
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