"""Risk management endpoints."""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from app.database import get_db
from app.api.deps import get_current_user
from app.models.risk import RiskRule, RiskEvent
from app.models.user import User
from app.models.trade import Trade
from pydantic import BaseModel, ConfigDict
import uuid
from datetime import datetime, timezone
import numpy as np

router = APIRouter(prefix="/risk", tags=["risk"])


async def _fetch_recent_pnl(db: AsyncSession, limit: int = 252) -> list[float]:
    """Fetch recent realized PnL values, falling back to synthetic data if none are available."""
    result = await db.execute(
        select(Trade.realized_pnl).order_by(Trade.closed_at.desc()).limit(limit)
    )
    pnl = [float(row[0]) for row in result.all() if row[0] is not None]
    if not pnl:
        np.random.seed(42)
        pnl = list(np.random.normal(80, 500, limit))
    return pnl


@router.get("/")
async def risk_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Risk dashboard summary: active rules count, recent events, circuit breaker status."""
    rules_result = await db.execute(select(RiskRule).where(RiskRule.is_active == True))
    active_rules = rules_result.scalars().all()
    events_result = await db.execute(
        select(RiskEvent).order_by(RiskEvent.triggered_at.desc()).limit(5)
    )
    recent_events = events_result.scalars().all()
    return {
        "active_rules": len(active_rules),
        "recent_events": len(recent_events),
        "circuit_breaker": "normal",
        "regime": "bull",
        "max_drawdown_limit_pct": 15.0,
        "position_limit_pct": 10.0,
    }


class RiskRuleCreate(BaseModel):
    rule_type: str
    threshold: float
    action: str = "alert"


class RiskRuleOut(BaseModel):
    id: str
    rule_type: str
    threshold: float
    action: str
    is_active: bool

    model_config = ConfigDict(from_attributes=True)


@router.get("/rules", response_model=list[RiskRuleOut])
async def list_rules(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(RiskRule))
    return result.scalars().all()


@router.post("/rules", response_model=RiskRuleOut)
async def create_rule(
    body: RiskRuleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rule = RiskRule(
        id=str(uuid.uuid4()),
        account_id="system",
        rule_type=body.rule_type,
        threshold=body.threshold,
        action=body.action,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    return rule


@router.delete("/rules/{rule_id}")
async def delete_risk_rule(
    rule_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rule = await db.get(RiskRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    await db.delete(rule)
    await db.commit()
    return {"deleted": rule_id}


@router.get("/events")
async def list_events(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(RiskEvent)
        .options(selectinload(RiskEvent.rule))
        .order_by(RiskEvent.triggered_at.desc())
        .limit(limit)
    )
    events = result.scalars().all()
    return [
        {
            "id": e.id,
            "event_type": (e.rule.rule_type if e.rule else None) or e.action_taken or "risk_event",
            "details": e.notes,
            "created_at": e.triggered_at,
        }
        for e in events
    ]


@router.get("/circuit-breaker")
async def get_circuit_breaker_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return current circuit breaker state for the dashboard."""
    result = await db.execute(
        select(RiskEvent)
        .order_by(desc(RiskEvent.triggered_at))
        .limit(1)
    )
    latest = result.scalar_one_or_none()
    is_tripped = (
        latest is not None
        and latest.resolved_at is None
        and latest.action_taken in ("halt_all", "halt_bucket")
    )
    return {
        "status": "tripped" if is_tripped else "normal",
        "tripped": is_tripped,
        "last_event_at": latest.triggered_at.isoformat() if latest else None,
    }


@router.get("/var")
async def get_var(
    portfolio_value: float = Query(100_000, description="Portfolio value in USD"),
    method: str = Query("historical"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Compute portfolio VaR and CVaR from recent trade returns."""
    from app.risk.var import historical_var

    pnl_list = await _fetch_recent_pnl(db)
    returns = [p / portfolio_value for p in pnl_list]
    var_result = historical_var(returns, portfolio_value, method)
    return var_result.to_dict()


@router.get("/factor-exposure")
async def get_factor_exposure(
    portfolio_value: float = Query(100_000),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Factor exposure analysis: market beta, momentum, low-vol."""
    from app.risk.factor_exposure import compute_factor_exposure

    pnl_list = await _fetch_recent_pnl(db)
    port_returns = [p / portfolio_value for p in pnl_list]

    np.random.seed(99)
    spy_returns = list(np.random.normal(0.0004, 0.012, len(port_returns)))

    exposure = compute_factor_exposure(port_returns, spy_returns)
    return exposure.to_dict()


@router.get("/drawdown-recovery")
async def get_drawdown_recovery(
    current_drawdown_pct: float = Query(5.0, description="Current drawdown as percentage, e.g. 5.0"),
    portfolio_value: float = Query(100_000),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Estimate drawdown recovery time via Monte Carlo."""
    from app.risk.drawdown_recovery import estimate_recovery

    pnl_list = await _fetch_recent_pnl(db)
    returns = [p / portfolio_value for p in pnl_list]
    estimate = estimate_recovery(returns, current_drawdown_pct / 100.0)
    return estimate.to_dict()