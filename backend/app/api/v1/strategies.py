"""Strategy management endpoints."""
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db, AsyncSessionLocal
from app.api.deps import get_current_user, get_current_active_superuser
from app.models.strategy import Strategy
from app.models.user import User
from app.strategies import STRATEGY_REGISTRY, list_desks, strategies_by_desk
from pydantic import BaseModel, ConfigDict

router = APIRouter(prefix="/strategies", tags=["strategies"])

logger = logging.getLogger(__name__)


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
    start = time.perf_counter()
    schema = {}
    for name, cls in STRATEGY_REGISTRY.items():
        if hasattr(cls, "DEFAULT_PARAMS"):
            schema[name] = {
                "params": cls.DEFAULT_PARAMS,
                "display_name": getattr(cls, "display_name", name),
            }
    elapsed = time.perf_counter() - start
    logger.info(
        "params_schema_fetched",
        endpoint="/strategies/params-schema",
        method="GET",
        user_id=getattr(current_user, "id", None),
        signal_count=len(schema),
        execution_time_ms=elapsed * 1000,
        pnl=None,
    )
    return schema


@router.get("/available")
async def list_available(current_user: User = Depends(get_current_user)):
    """List all registered strategy classes."""
    start = time.perf_counter()
    result = [{"name": k} for k in STRATEGY_REGISTRY.keys()]
    elapsed = time.perf_counter() - start
    logger.info(
        "available_strategies_listed",
        endpoint="/strategies/available",
        method="GET",
        user_id=getattr(current_user, "id", None),
        signal_count=len(result),
        execution_time_ms=elapsed * 1000,
        pnl=None,
    )
    return result


@router.get("/desks")
async def list_strategy_desks(current_user: User = Depends(get_current_user)):
    """Unified cross-desk view: every strategy grouped by trading desk.

    Desks are derived from each strategy's own attributes (no hand-maintained list),
    so the equities/crypto/options/prediction-market/TradingView desks all share one
    format and a new strategy is placed automatically.
    """
    start = time.perf_counter()
    grouped = strategies_by_desk()
    response = {
        "desks": list_desks(),
        "by_desk": grouped,
        "counts": {desk: len(members) for desk, members in grouped.items()},
        "total": sum(len(m) for m in grouped.values()),
    }
    elapsed = time.perf_counter() - start
    logger.info(
        "strategy_desks_fetched",
        endpoint="/strategies/desks",
        method="GET",
        user_id=getattr(current_user, "id", None),
        signal_count=response["total"],
        execution_time_ms=elapsed * 1000,
        pnl=None,
    )
    return response


@router.get("/active")
async def list_active(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Return the strategies that are currently running in the strategy runner.

    Reads from app.state.active_strategies (populated at startup by main.py).
    Falls back to querying the DB when app state is not yet populated.
    """
    start = time.perf_counter()
    # Try in-process state first (populated by lifespan at startup)
    active = getattr(request.app.state, "active_strategies", None)
    if active is not None:
        elapsed = time.perf_counter() - start
        logger.info(
            "active_strategies_fetched",
            endpoint="/strategies/active",
            method="GET",
            user_id=getattr(current_user, "id", None),
            signal_count=len(active) if hasattr(active, "__len__") else 0,
            execution_time_ms=elapsed * 1000,
            pnl=None,
        )
        return active

    # Fallback: query DB directly with a lightweight column selection
    try:
        async with AsyncSessionLocal() as db:
            stmt = select(
                Strategy.name,
                Strategy.symbols,
                Strategy.tick_interval_seconds,
                Strategy.confidence_threshold,
            ).where(Strategy.is_enabled.is_(True))
            result = await db.execute(stmt)
            rows = result.mappings().all()
            data = [
                {
                    "name": row["name"],
                    "symbols": row["symbols"] if isinstance(row["symbols"], list) else [],
                    "tick_interval_seconds": int(row.get("tick_interval_seconds", 3600)),
                    "confidence_threshold": float(row.get("confidence_threshold", 0.6)),
                    "is_running": True,
                }
                for row in rows
            ]
            elapsed = time.perf_counter() - start
            logger.info(
                "active_strategies_fetched_db",
                endpoint="/strategies/active",
                method="GET",
                user_id=getattr(current_user, "id", None),
                signal_count=len(data),
                execution_time_ms=elapsed * 1000,
                pnl=None,
            )
            return data
    except Exception:
        elapsed = time.perf_counter() - start
        logger.info(
            "active_strategies_error",
            endpoint="/strategies/active",
            method="GET",
            user_id=getattr(current_user, "id", None),
            signal_count=0,
            execution_time_ms=elapsed * 1000,
            pnl=None,
        )
        # Return empty list rather than crashing — frontend must handle this gracefully
        return []


@router.get("/", response_model=list[StrategyOut])
async def list_strategies(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start = time.perf_counter()
    result = await db.execute(select(Strategy))
    strategies = result.scalars().all()
    elapsed = time.perf_counter() - start
    logger.info(
        "strategies_listed",
        endpoint="/strategies/",
        method="GET",
        user_id=getattr(current_user, "id", None),
        signal_count=len(strategies),
        execution_time_ms=elapsed * 1000,
        pnl=None,
    )
    return strategies


@router.patch("/{strategy_id}/toggle")
async def toggle_strategy(
    strategy_id: str,
    body: StrategyToggle,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_superuser),
):
    start = time.perf_counter()
    result = await db.execute(select(Strategy).where(Strategy.id == strategy_id))
    strategy = result.scalar_one_or_none()
    if not strategy:
        elapsed = time.perf_counter() - start
        logger.info(
            "strategy_toggle_failed",
            endpoint=f"/strategies/{strategy_id}/toggle",
            method="PATCH",
            user_id=getattr(current_user, "id", None),
            signal_count=0,
            execution_time_ms=elapsed * 1000,
            pnl=None,
            reason="not_found",
        )
        raise HTTPException(404, "Strategy not found")
    strategy.is_enabled = body.is_enabled
    await db.commit()
    elapsed = time.perf_counter() - start
    logger.info(
        "strategy_toggled",
        endpoint=f"/strategies/{strategy_id}/toggle",
        method="PATCH",
        user_id=getattr(current_user, "id", None),
        signal_count=1,
        execution_time_ms=elapsed * 1000,
        pnl=None,
        new_state=body.is_enabled,
    )
    return {"id": strategy_id, "is_enabled": body.is_enabled}