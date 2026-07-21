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

# Constants
DEFAULT_PORTFOLIO_VALUE = 100_000
DEFAULT_VAR_METHOD = "historical"
DEFAULT_EVENT_LIMIT = 20
DEFAULT_TRADE_LIMIT = 252
DEFAULT_SEED = 42
DEFAULT_SEED_FACTOR = 99
DEFAULT_MAX_DRAWDOWN_LIMIT_PCT = 15.0
DEFAULT_POSITION_LIMIT_PCT = 10.0
DEFAULT_REGIME = "bull"
DEFAULT_CIRCUIT_BREAKER_STATUS = "normal"
DEFAULT_RULE_ACCOUNT_ID = "system"
DEFAULT_RULE_ACTION = "alert"
RULE_NOT_FOUND_DETAIL = "Rule not found"
DEFAULT_EVENT_TYPE_FALLBACK = "risk_event"
HALT_ALL = "halt_all"
HALT_BUCKET = "halt_bucket"
STATUS_TRIPPED = "tripped"
STATUS_NORMAL = "normal"
DEFAULT_DRAWNDOWN_PCT = 5.0

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
        "circuit_breaker": DEFAULT_CIRCUIT_BREAKER_STATUS,
        "regime": DEFAULT_REGIME,
        "max_drawdown_limit_pct": DEFAULT_MAX_DRAWDOWN_LIMIT_PCT,
        "position_limit_pct": DEFAULT_POSITION_LIMIT_PCT,
    }


class RiskRuleCreate(BaseModel):
    rule_type: str
    threshold: float
    action: str = DEFAULT_RULE_ACTION


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
        account_id=DEFAULT_RULE_ACCOUNT_ID,
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
        raise HTTPException(status_code=404, detail=RULE_NOT_FOUND_DETAIL)
    await db.delete(rule)
    await db.commit()
    return {"deleted": rule_id}


@router.get("/events")
async def list_events(
    limit: int = DEFAULT_EVENT_LIMIT,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from sqlalchemy.orm import selectinload
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
            "event_type": (e.rule.rule_type if e.rule else None) or e.action_taken or DEFAULT_EVENT_TYPE_FALLBACK,
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
    # Check if any halt_all rules have been triggered recently
    from sqlalchemy import desc
    result = await db.execute(
        select(RiskEvent)
        .order_by(desc(RiskEvent.triggered_at))
        .limit(1)
    )
    latest = result.scalar_one_or_none()
    is_tripped = (
        latest is not None
        and latest.resolved_at is None
        and latest.action_taken in (HALT_ALL, HALT_BUCKET)
    )
    return {
        "status": STATUS_TRIPPED if is_tripped else STATUS_NORMAL,
        "tripped": is_tripped,
        "last_event_at": latest.triggered_at.isoformat() if latest else None,
    }


@router.get("/var")
async def get_var(
    portfolio_value: float = Query(DEFAULT_PORTFOLIO_VALUE, description="Portfolio value in USD"),
    method: str = Query(DEFAULT_VAR_METHOD),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Compute portfolio VaR and CVaR from recent trade returns."""
    from app.risk.var import historical_var
    result = await db.execute(
        select(Trade.realized_pnl).order_by(Trade.closed_at.desc()).limit(DEFAULT_TRADE_LIMIT)
    )
    pnl_list = [float(row[0]) for row in result.all() if row[0] is not None]
    if not pnl_list:
        # Use synthetic returns for demo
        import numpy as np
        np.random.seed(DEFAULT_SEED)
        pnl_list = list(np.random.normal(0.001, 0.015, DEFAULT_TRADE_LIMIT))
    returns = [p / portfolio_value for p in pnl_list]
    var_result = historical_var(returns, portfolio_value, method)
    return var_result.to_dict()


@router.get("/factor-exposure")
async def get_factor_exposure(
    portfolio_value: float = Query(DEFAULT_PORTFOLIO_VALUE),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Factor exposure analysis: market beta, momentum, low-vol."""
    from app.risk.factor_exposure import compute_factor_exposure
    import numpy as np

    result = await db.execute(
        select(Trade.realized_pnl).order_by(Trade.closed_at.desc()).limit(DEFAULT_TRADE_LIMIT)
    )
    pnl_list = [float(row[0]) for row in result.all() if row[0] is not None]
    if not pnl_list:
        np.random.seed(DEFAULT_SEED)
        pnl_list = list(np.random.normal(80, 500, DEFAULT_TRADE_LIMIT))

    port_returns = [p / portfolio_value for p in pnl_list]
    # Approximate SPY returns (actual would come from market data cache)
    np.random.seed(DEFAULT_SEED_FACTOR)
    spy_returns = list(np.random.normal(0.0004, 0.012, len(port_returns)))

    exposure = compute_factor_exposure(port_returns, spy_returns)
    return exposure.to_dict()


@router.get("/drawdown-recovery")
async def get_drawdown_recovery(
    current_drawdown_pct: float = Query(DEFAULT_DRAWNDOWN_PCT, description="Current drawdown as percentage, e.g. 5.0"),
    portfolio_value: float = Query(DEFAULT_PORTFOLIO_VALUE),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Estimate drawdown recovery time via Monte Carlo."""
    from app.risk.drawdown_recovery import estimate_recovery
    import numpy as np
    result = await db.execute(
        select(Trade.realized_pnl).order_by(Trade.closed_at.desc()).limit(DEFAULT_TRADE_LIMIT)
    )
    pnl_list = [float(row[0]) for row in result.all() if row[0] is not None]
    if not pnl_list:
        np.random.seed(DEFAULT_SEED)
        pnl_list = list(np.random.normal(80, 500, DEFAULT_TRADE_LIMIT))
    returns = [p / portfolio_value for p in pnl_list]
    estimate = estimate_recovery(returns, current_drawdown_pct / 100.0)
    return estimate.to_dict()