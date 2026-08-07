"""Strategy management endpoints."""
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_superuser, get_current_user
from app.database import AsyncSessionLocal, get_db
from app.models.strategy import Strategy
from app.models.user import User
from app.strategies import STRATEGY_REGISTRY, list_desks, strategies_by_desk
from pydantic import BaseModel, ConfigDict, validator

router = APIRouter(prefix="/strategies", tags=["strategies"])

logger = logging.getLogger(__name__)


def _log_metrics(
    endpoint: str,
    start_time: float,
    count: Optional[int] = None,
    pnl: Optional[float] = None,
) -> None:
    """Log structured metrics for an endpoint."""
    duration = time.perf_counter() - start_time
    logger.info(
        f"{endpoint} completed",
        extra={
            "endpoint": endpoint,
            "duration_seconds": round(duration, 4),
            "record_count": count,
            "pnl": pnl,
        },
    )


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

    @validator("is_enabled")
    def must_be_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError("is_enabled must be a boolean")
        return v


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
    _log_metrics("get_params_schema", start, count=len(schema))
    return schema


@router.get("/available")
async def list_available(current_user: User = Depends(get_current_user)):
    """List all registered strategy classes."""
    start = time.perf_counter()
    result = [{"name": k} for k in STRATEGY_REGISTRY.keys()]
    _log_metrics("list_available", start, count=len(result))
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
    _log_metrics("list_strategy_desks", start, count=response["total"])
    return response


def _get_active_from_state(app) -> Optional[List[Dict[str, Any]]]:
    """Retrieve active strategies from the in‑process state if available."""
    return getattr(app.state, "active_strategies", None)


def _process_strategy_rows(
    rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Validate and transform raw DB rows into the API response format."""
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        tick = row.get("tick_interval_seconds", 3600)
        conf = row.get("confidence_threshold", 0.6)
        # Additional sanity checks
        if tick <= 0 or not (0.0 <= conf <= 1.0):
            continue
        filtered.append(
            {
                "name": row["name"],
                "symbols": row["symbols"] if isinstance(row["symbols"], list) else [],
                "tick_interval_seconds": int(tick),
                "confidence_threshold": float(conf),
                "is_running": True,
            }
        )
    return filtered


async def _fetch_active_strategies_from_db() -> List[Dict[str, Any]]:
    """Query the database for enabled strategies with a minimum confidence threshold."""
    try:
        async with AsyncSessionLocal() as db:
            stmt = (
                select(
                    Strategy.name,
                    Strategy.symbols,
                    Strategy.tick_interval_seconds,
                    Strategy.confidence_threshold,
                )
                .where(Strategy.is_enabled.is_(True))
                .where(Strategy.confidence_threshold >= 0.7)
            )
            result = await db.execute(stmt)
            rows = result.mappings().all()
            return _process_strategy_rows(rows)
    except Exception:
        # Return empty list rather than crashing — frontend must handle this gracefully
        return []


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
    active = _get_active_from_state(request.app)
    if active is not None:
        _log_metrics("list_active", start, count=len(active))
        return active
    active = await _fetch_active_strategies_from_db()
    _log_metrics("list_active", start, count=len(active))
    return active


@router.get("/", response_model=list[StrategyOut])
async def list_strategies(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start = time.perf_counter()
    result = await db.execute(select(Strategy))
    strategies = result.scalars().all()
    _log_metrics("list_strategies", start, count=len(strategies))
    return strategies


@router.patch("/{strategy_id}/toggle")
async def toggle_strategy(
    strategy_id: str,
    body: StrategyToggle,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_superuser),
):
    """Enable or disable a strategy, enforcing a minimum confidence threshold for activation."""
    start = time.perf_counter()
    result = await db.execute(select(Strategy).where(Strategy.id == strategy_id))
    strategy = result.scalar_one_or_none()
    if not strategy:
        _log_metrics("toggle_strategy", start, count=0)
        raise HTTPException(status_code=404, detail="Strategy not found")
    if body.is_enabled and strategy.confidence_threshold < 0.7:
        _log_metrics("toggle_strategy", start, count=0)
        raise HTTPException(
            status_code=400,
            detail=(
                "Cannot enable strategy: confidence_threshold "
                f"{strategy.confidence_threshold:.2f} is below the required minimum of 0.70."
            ),
        )
    strategy.is_enabled = body.is_enabled
    await db.commit()
    _log_metrics("toggle_strategy", start, count=1)
    return {"id": strategy_id, "is_enabled": body.is_enabled}