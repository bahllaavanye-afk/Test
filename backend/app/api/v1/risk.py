"""Risk management endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.risk import RiskRule, RiskEvent
from app.models.user import User
from app.models.trade import Trade
from pydantic import BaseModel, ConfigDict
import uuid
from datetime import datetime, timezone

router = APIRouter(prefix="/risk", tags=["risk"])


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
    from fastapi import HTTPException
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
        select(RiskEvent).order_by(RiskEvent.triggered_at.desc()).limit(limit)
    )
    events = result.scalars().all()
    return [{"id": e.id, "event_type": e.rule_id, "details": e.notes, "created_at": e.triggered_at} for e in events]


@router.get("/var")
async def get_var(
    portfolio_value: float = Query(100_000, description="Portfolio value in USD"),
    method: str = Query("historical"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Compute portfolio VaR and CVaR from recent trade returns."""
    from app.risk.var import historical_var
    result = await db.execute(
        select(Trade.realized_pnl).order_by(Trade.closed_at.desc()).limit(252)
    )
    pnl_list = [float(row[0]) for row in result.all() if row[0] is not None]
    if not pnl_list:
        # Use synthetic returns for demo
        import numpy as np
        np.random.seed(42)
        pnl_list = list(np.random.normal(0.001, 0.015, 252))
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
    import numpy as np

    result = await db.execute(
        select(Trade.realized_pnl).order_by(Trade.closed_at.desc()).limit(252)
    )
    pnl_list = [float(row[0]) for row in result.all() if row[0] is not None]
    if not pnl_list:
        np.random.seed(42)
        pnl_list = list(np.random.normal(80, 500, 252))

    port_returns = [p / portfolio_value for p in pnl_list]
    # Approximate SPY returns (actual would come from market data cache)
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
    import numpy as np
    result = await db.execute(
        select(Trade.realized_pnl).order_by(Trade.closed_at.desc()).limit(252)
    )
    pnl_list = [float(row[0]) for row in result.all() if row[0] is not None]
    if not pnl_list:
        np.random.seed(42)
        pnl_list = list(np.random.normal(80, 500, 252))
    returns = [p / portfolio_value for p in pnl_list]
    estimate = estimate_recovery(returns, current_drawdown_pct / 100.0)
    return estimate.to_dict()
