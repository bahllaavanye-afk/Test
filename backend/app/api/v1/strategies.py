"""Strategy management endpoints."""
from datetime import UTC, datetime, timedelta

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
                    "tick_interval_seconds": int(getattr(s, "tick_interval_seconds", 3600)),
                    "confidence_threshold": float(getattr(s, "confidence_threshold", 0.6)),
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


@router.get("/dashboard")
async def bot_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Option Alpha-style bot dashboard: per-strategy P&L, win rate, allocation,
    risk, open positions, and today's change — all in one call.
    """
    strategies_result = await db.execute(select(Strategy))
    strategies = strategies_result.scalars().all()

    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    # Aggregate all-time stats per strategy_id
    all_time = await db.execute(
        select(
            Trade.strategy_id,
            func.sum(Trade.realized_pnl).label("total_pnl"),
            func.count(Trade.id).label("total_trades"),
            func.sum(
                func.cast(Trade.realized_pnl > 0, type_=func.count(Trade.id).__class__)
            ).label("wins"),
        ).group_by(Trade.strategy_id)
    )
    all_time_rows = {r.strategy_id: r for r in all_time}

    # Today's P&L per strategy_id
    today = await db.execute(
        select(
            Trade.strategy_id,
            func.sum(Trade.realized_pnl).label("today_pnl"),
        )
        .where(Trade.closed_at >= today_start)
        .group_by(Trade.strategy_id)
    )
    today_rows = {r.strategy_id: r for r in today}

    # Win count workaround (SQLAlchemy-safe)
    wins_result = await db.execute(
        select(
            Trade.strategy_id,
            func.count(Trade.id).label("win_count"),
        )
        .where(Trade.realized_pnl > 0)
        .group_by(Trade.strategy_id)
    )
    wins_map = {r.strategy_id: r.win_count for r in wins_result}

    # Vol targeting scalars if available
    try:
        from app.risk.vol_targeting import vol_targeter
        vol_stats = {s["strategy_key"]: s for s in vol_targeter.get_all_stats()}
    except Exception:
        vol_stats = {}

    bots = []
    total_pnl = 0.0
    total_allocation = 0.0
    total_risk = 0.0

    for s in strategies:
        at = all_time_rows.get(s.id)
        tod = today_rows.get(s.id)

        total_trades = at.total_trades if at else 0
        strategy_pnl = float(at.total_pnl) if at and at.total_pnl else 0.0
        today_pnl = float(tod.today_pnl) if tod and tod.today_pnl else 0.0
        wins = wins_map.get(s.id, 0)
        win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else None

        allocation = float(s.params.get("allocation_usd", 2500.0)) if s.params else 2500.0
        risk_usd = float(s.params.get("risk_usd", 0.0)) if s.params else 0.0
        return_pct = round(strategy_pnl / allocation * 100, 2) if allocation > 0 else 0.0
        today_change_pct = round(today_pnl / allocation * 100, 3) if allocation > 0 else 0.0

        vs = vol_stats.get(f"{s.name}_{s.symbols[0] if s.symbols else ''}")
        vol_scalar = vs["scalar"] if vs else None

        bots.append({
            "id": s.id,
            "name": s.name,
            "display_name": s.display_name or s.name.replace("_", " ").title(),
            "market_type": s.market_type,
            "strategy_type": s.strategy_type,
            "risk_bucket": s.risk_bucket,
            "is_enabled": s.is_enabled,
            "symbols": s.symbols,
            "total_pnl": round(strategy_pnl, 2),
            "return_pct": return_pct,
            "today_pnl": round(today_pnl, 2),
            "today_change_pct": today_change_pct,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "allocation": allocation,
            "risk_usd": risk_usd,
            "vol_scalar": vol_scalar,
            "confidence_threshold": s.confidence_threshold,
            "tick_interval_seconds": s.tick_interval_seconds,
        })
        total_pnl += strategy_pnl
        total_allocation += allocation
        total_risk += risk_usd

    all_wins = sum(wins_map.get(s.id, 0) for s in strategies)
    all_trades = sum((all_time_rows[s.id].total_trades if s.id in all_time_rows else 0) for s in strategies)

    return {
        "summary": {
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(total_pnl / total_allocation * 100, 2) if total_allocation > 0 else 0.0,
            "today_pnl": sum(b["today_pnl"] for b in bots),
            "today_change_pct": round(sum(b["today_pnl"] for b in bots) / total_allocation * 100, 3) if total_allocation > 0 else 0.0,
            "total_risk": round(total_risk, 2),
            "total_allocation": round(total_allocation, 2),
            "total_bots": len(bots),
            "active_bots": sum(1 for b in bots if b["is_enabled"]),
            "overall_win_rate": round(all_wins / all_trades * 100, 1) if all_trades > 0 else None,
        },
        "bots": bots,
    }


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
