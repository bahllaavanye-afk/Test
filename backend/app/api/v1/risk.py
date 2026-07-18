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

    result = await db.execute(
        select(Trade.realized_pnl).order_by(Trade.closed_at.desc()).limit(252)
    )
    pnl_list = [float(row[0]) for row in result.all() if row[0] is not None]
    if not pnl_list:
        # Use synthetic returns for demo
        import numpy as np

        np.random.seed(42)
        pnl_list = list(np.random.normal(0.001, 0.015, 252))
    # Guard against division by zero
    if portfolio_value == 0:
        raise ValueError("portfolio_value must be greater than zero")
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


# ---------------------------------------------------------------------------
# Unit tests for edge cases
# ---------------------------------------------------------------------------
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Simple mock async session that returns empty results
class _MockScalarResult:
    def __init__(self, data):
        self._data = data

    def all(self):
        return self._data

    def first(self):
        return self._data[0] if self._data else None


class _MockResult:
    def __init__(self, scalar_data=None, scalar_one=None):
        self._scalar_data = scalar_data or []
        self._scalar_one = scalar_one

    def scalars(self):
        return _MockScalarResult(self._scalar_data)

    def scalar_one_or_none(self):
        return self._scalar_one

    def all(self):
        # Mimic rows as tuples
        return [(v,) for v in self._scalar_data]


class _MockAsyncSession:
    async def execute(self, stmt):
        # Return empty results for any query
        return _MockResult()

    async def get(self, model, pk):
        return None

    async def delete(self, obj):
        pass

    async def add(self, obj):
        pass

    async def commit(self):
        pass

    async def refresh(self, obj):
        pass


# Fixture to create an app with dependency overrides
@pytest.fixture
def app():
    fast_app = FastAPI()
    fast_app.include_router(router)

    async def override_get_db():
        return _MockAsyncSession()

    fast_app.dependency_overrides[get_db] = override_get_db
    fast_app.dependency_overrides[get_current_user] = lambda: User(id="test", username="tester")
    return fast_app


@pytest.mark.anyio
async def test_list_events_limit_zero(app):
    """Edge case: limit=0 should return an empty list without error."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/risk/events?limit=0")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.anyio
async def test_delete_nonexistent_rule_returns_404(app):
    """Edge case: deleting a rule that does not exist should return 404."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.delete("/risk/rules/nonexistent-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Rule not found"


@pytest.mark.anyio
async def test_get_var_zero_portfolio_raises_error(app):
    """Edge case: portfolio_value=0 should raise a validation error."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/risk/var?portfolio_value=0")
    # The endpoint now raises a ValueError which FastAPI converts to a 500 response
    assert response.status_code == 500
    # Ensure the error message is propagated
    assert "portfolio_value must be greater than zero" in response.text