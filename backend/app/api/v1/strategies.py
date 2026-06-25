"""Strategy management endpoints."""
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_superuser, get_current_user
from app.database import get_db
from app.models.strategy import Strategy
from app.models.trade import Trade
from app.models.user import User
from app.strategies import STRATEGY_REGISTRY

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

    # Fallback: query DB directly
    try:
        from app.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Strategy).where(Strategy.is_enabled == True)  # noqa: E712
            )
            rows = result.scalars().all()
            return [
                {
                    "name": s.name,
                    "symbols": s.symbols if isinstance(s.symbols, list) else [],
                    "tick_interval_seconds": int(
                        getattr(s, "tick_interval_seconds", 3600)
                    ),
                    "confidence_threshold": float(
                        getattr(s, "confidence_threshold", 0.6)
                    ),
                    "is_running": True,
                }
                for s in rows
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


# --------------------------------------------------------------------------- #
# Helper functions for bot_dashboard (refactored for readability)
# --------------------------------------------------------------------------- #


async def _fetch_all_time_stats(db: AsyncSession) -> dict[int, any]:
    """Fetch all‑time P&L, trade count and win count per strategy."""
    result = await db.execute(
        select(
            Trade.strategy_id,
            func.sum(Trade.realized_pnl).label("total_pnl"),
            func.count(Trade.id).label("total_trades"),
            func.sum(
                func.cast(
                    Trade.realized_pnl > 0,
                    type_=func.count(Trade.id).__class__,
                )
            ).label("wins"),
        ).group_by(Trade.strategy_id)
    )
    rows = result.scalars().all()
    return {r.strategy_id: r for r in rows}


async def _fetch_today_stats(db: AsyncSession, today_start: datetime) -> dict[int, any]:
    """Fetch today's P&L per strategy."""
    result = await db.execute(
        select(
            Trade.strategy_id,
            func.sum(Trade.realized_pnl).label("today_pnl"),
        )
        .where(Trade.closed_at >= today_start)
        .group_by(Trade.strategy_id)
    )
    rows = result.scalars().all()
    return {r.strategy_id: r for r in rows}


async def _fetch_wins_map(db: AsyncSession) -> dict[int, int]:
    """Fetch total win count per strategy (realized P&L > 0)."""
    result = await db.execute(
        select(
            Trade.strategy_id,
            func.count(Trade.id).label("win_count"),
        )
        .where(Trade.realized_pnl > 0)
        .group_by(Trade.strategy_id)
    )
    rows = result.scalars().all()
    return {r.strategy_id: r.win_count for r in rows}


def _load_vol_stats() -> dict[str, dict]:
    """Load volatility targeting stats if available."""
    try:
        from app.risk.vol_targeting import vol_targeter

        return {s["strategy_key"]: s for s in vol_targeter.get_all_stats()}
    except Exception:
        return {}


def _build_bot_entry(
    strategy: Strategy,
    all_time_row: any,
    today_row: any,
    wins: int,
    vol_stats: dict[str, dict],
) -> dict:
    """Assemble a single bot dictionary for the dashboard response."""
    total_trades = all_time_row.total_trades if all_time_row else 0
    strategy_pnl = float(all_time_row.total_pnl) if all_time_row and all_time_row.total_pnl else 0.0
    today_pnl = float(today_row.today_pnl) if today_row and today_row.today_pnl else 0.0
    win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else None

    allocation = float(strategy.params.get("allocation_usd", 2500.0)) if strategy.params else 2500.0
    risk_usd = float(strategy.params.get("risk_usd", 0.0)) if strategy.params else 0.0
    return_pct = round(strategy_pnl / allocation * 100, 2) if allocation > 0 else 0.0
    today_change_pct = round(today_pnl / allocation * 100, 3) if allocation > 0 else 0.0

    vol_key = f"{strategy.name}_{strategy.symbols[0] if strategy.symbols else ''}"
    vol_scalar = vol_stats.get(vol_key, {}).get("scalar")

    return {
        "id": strategy.id,
        "name": strategy.name,
        "display_name": strategy.display_name
        or strategy.name.replace("_", " ").title(),
        "market_type": strategy.market_type,
        "strategy_type": strategy.strategy_type,
        "risk_bucket": strategy.risk_bucket,
        "is_enabled": strategy.is_enabled,
        "symbols": strategy.symbols,
        "total_pnl": round(strategy_pnl, 2),
        "return_pct": return_pct,
        "today_pnl": round(today_pnl, 2),
        "today_change_pct": today_change_pct,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "allocation": allocation,
        "risk_usd": risk_usd,
        "vol_scalar": vol_scalar,
        "confidence_threshold": strategy.confidence_threshold,
        "tick_interval_seconds": strategy.tick_interval_seconds,
    }


def _compute_summary(
    bots: list[dict],
    total_pnl: float,
    total_allocation: float,
    total_risk: float,
    all_wins: int,
    all_trades: int,
) -> dict:
    """Calculate aggregate statistics for the dashboard."""
    summary = {
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / total_allocation * 100, 2)
        if total_allocation > 0
        else 0.0,
        "today_pnl": sum(b["today_pnl"] for b in bots),
        "today_change_pct": round(
            sum(b["today_pnl"] for b in bots) / total_allocation * 100, 3
        )
        if total_allocation > 0
        else 0.0,
        "total_risk": round(total_risk, 2),
        "total_allocation": round(total_allocation, 2),
        "total_bots": len(bots),
        "active_bots": sum(1 for b in bots if b["is_enabled"]),
        "overall_win_rate": round(all_wins / all_trades * 100, 1)
        if all_trades > 0
        else None,
    }
    return summary


@router.get("/dashboard")
async def bot_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Option Alpha‑style bot dashboard: per‑strategy P&L, win rate, allocation,
    risk, open positions, and today's change — all in one call.
    """
    # Load strategies
    strategies_result = await db.execute(select(Strategy))
    strategies = strategies_result.scalars().all()

    # Reference time for "today"
    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    # Fetch aggregated DB metrics
    all_time_rows = await _fetch_all_time_stats(db)
    today_rows = await _fetch_today_stats(db, today_start)
    wins_map = await _fetch_wins_map(db)
    vol_stats = _load_vol_stats()

    bots = []
    total_pnl = 0.0
    total_allocation = 0.0
    total_risk = 0.0

    for strategy in strategies:
        at = all_time_rows.get(strategy.id)
        td = today_rows.get(strategy.id)
        wins = wins_map.get(strategy.id, 0)

        bot_entry = _build_bot_entry(strategy, at, td, wins, vol_stats)
        bots.append(bot_entry)

        total_pnl += bot_entry["total_pnl"]
        total_allocation += bot_entry["allocation"]
        total_risk += bot_entry["risk_usd"]

    all_wins = sum(wins_map.get(s.id, 0) for s in strategies)
    all_trades = sum(
        all_time_rows[s.id].total_trades if s.id in all_time_rows else 0 for s in strategies
    )

    summary = _compute_summary(
        bots, total_pnl, total_allocation, total_risk, all_wins, all_trades
    )

    return {"summary": summary, "bots": bots}