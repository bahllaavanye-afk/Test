"""Strategy management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db, AsyncSessionLocal
from app.api.deps import get_current_user, get_current_active_superuser
from app.models.strategy import Strategy
from app.models.user import User
from app.strategies import STRATEGY_REGISTRY, desk_of, list_desks, strategies_by_desk
from pydantic import BaseModel, ConfigDict, Field, validator
import uuid

router = APIRouter(prefix="/strategies", tags=["strategies"])


class StrategyOut(BaseModel):
    """Schema representing a strategy as returned by the API."""

    id: str = Field(
        ...,
        description="Unique identifier of the strategy (UUID string).",
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    name: str = Field(
        ...,
        description="Human‑readable name of the strategy.",
        example="Mean Reversion 20",
    )
    market_type: str = Field(
        ...,
        description="Market classification (e.g., equities, crypto, options).",
        example="equities",
    )
    strategy_type: str = Field(
        ...,
        description="Technical classification of the strategy.",
        example="mean_rev_20_1.5",
    )
    risk_bucket: str = Field(
        ...,
        description="Risk bucket the strategy belongs to.",
        example="high",
    )
    is_enabled: bool = Field(
        ...,
        description="Indicates whether the strategy is currently enabled.",
        example=True,
    )
    symbols: list[str] = Field(
        ...,
        description="List of ticker symbols the strategy trades.",
        example=["AAPL", "MSFT"],
    )
    tick_interval_seconds: float = Field(
        ...,
        description="Frequency, in seconds, at which the strategy ticks.",
        example=3600.0,
        gt=0,
    )
    confidence_threshold: float = Field(
        ...,
        description="Minimum confidence level required to execute a trade.",
        example=0.6,
        ge=0.0,
        le=1.0,
    )

    model_config = ConfigDict(from_attributes=True)

    @validator("id")
    def validate_uuid(cls, v: str) -> str:
        """Ensure the id is a valid UUID string."""
        try:
            uuid.UUID(v)
        except ValueError as exc:
            raise ValueError("id must be a valid UUID") from exc
        return v

    @validator("symbols", each_item=True)
    def validate_symbol(cls, v: str) -> str:
        """Each symbol must be a non‑empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("symbol must be a non‑empty string")
        return v

    @validator("tick_interval_seconds")
    def validate_tick_interval(cls, v: float) -> float:
        """Tick interval must be greater than zero."""
        if v <= 0:
            raise ValueError("tick_interval_seconds must be greater than zero")
        return v

    @validator("confidence_threshold")
    def validate_confidence(cls, v: float) -> float:
        """Confidence threshold must be between 0 and 1 inclusive."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v


class StrategyToggle(BaseModel):
    """Schema used to enable or disable a strategy."""

    is_enabled: bool = Field(
        ...,
        description="Flag indicating whether the strategy should be enabled.",
        example=True,
    )


@router.get("/params-schema")
async def get_params_schema(current_user: User = Depends(get_current_user)):
    """Return configurable params for each strategy that exposes DEFAULT_PARAMS."""
    schema = {}
    for name, cls in STRATEGY_REGISTRY.items():
        if hasattr(cls, "DEFAULT_PARAMS"):
            schema[name] = {
                "params": cls.DEFAULT_PARAMS,
                "display_name": getattr(cls, "display_name", name),
            }
    return schema


@router.get("/available")
async def list_available(current_user: User = Depends(get_current_user)):
    """List all registered strategy classes."""
    return [{"name": k} for k in STRATEGY_REGISTRY.keys()]


@router.get("/desks")
async def list_strategy_desks(current_user: User = Depends(get_current_user)):
    """Unified cross-desk view: every strategy grouped by trading desk.

    Desks are derived from each strategy's own attributes (no hand-maintained list),
    so the equities/crypto/options/prediction-market/TradingView desks all share one
    format and a new strategy is placed automatically.
    """
    grouped = strategies_by_desk()
    return {
        "desks": list_desks(),
        "by_desk": grouped,
        "counts": {desk: len(members) for desk, members in grouped.items()},
        "total": sum(len(m) for m in grouped.values()),
    }


@router.get("/active")
async def list_active(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Return the strategies that are currently running in the strategy runner.

    Reads from app.state.active_strategies (populated at startup by main.py).
    Falls back to querying the DB when app state is not yet populated.
    """
    # Try in-process state first (populated by lifespan at startup)
    active = getattr(request.app.state, "active_strategies", None)
    if active is not None:
        return active

    # Fallback: query DB directly with a lightweight column selection
    try:
        async with AsyncSessionLocal() as db:
            stmt = select(
                Strategy.name,
                Strategy.symbols,
                Strategy.tick_interval_seconds,
                Strategy.confidence_threshold,
            ).where(Strategy.is_enabled == True)  # noqa: E712
            result = await db.execute(stmt)
            rows = result.mappings().all()
            return [
                {
                    "name": row["name"],
                    "symbols": row["symbols"] if isinstance(row["symbols"], list) else [],
                    "tick_interval_seconds": int(row.get("tick_interval_seconds", 3600)),
                    "confidence_threshold": float(row.get("confidence_threshold", 0.6)),
                    "is_running": True,
                }
                for row in rows
            ]
    except Exception:
        # Return empty list rather than crashing — frontend must handle this gracefully
        return []


@router.get("/", response_model=list[StrategyOut])
async def list_strategies(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Strategy))
    return result.scalars().all()


@router.patch("/{strategy_id}/toggle")
async def toggle_strategy(
    strategy_id: str,
    body: StrategyToggle,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_superuser),
):
    result = await db.execute(select(Strategy).where(Strategy.id == strategy_id))
    strategy = result.scalar_one_or_none()
    if not strategy:
        raise HTTPException(404, "Strategy not found")
    strategy.is_enabled = body.is_enabled
    await db.commit()
    return {"id": strategy_id, "is_enabled": body.is_enabled}