"""Strategy management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db, AsyncSessionLocal
from app.api.deps import get_current_user, get_current_active_superuser
from app.models.strategy import Strategy
from app.models.user import User
from app.strategies import STRATEGY_REGISTRY, desk_of, list_desks, strategies_by_desk
from pydantic import BaseModel, ConfigDict
import uuid

router = APIRouter(prefix="/strategies", tags=["strategies"])


class StrategyOut(BaseModel):
    id: str
    name: str
    market_type: str
    strategy_type: str
    risk_bucket: str
    is_enabled: bool
    symbols: list[str]
    tick_interval_seconds: float
    confidence_threshold: float

    model_config = ConfigDict(from_attributes=True)


class StrategyToggle(BaseModel):
    is_enabled: bool


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