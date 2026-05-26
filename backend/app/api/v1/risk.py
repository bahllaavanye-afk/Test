"""Risk management endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.risk import RiskRule, RiskEvent
from app.models.user import User
from pydantic import BaseModel
import uuid
from datetime import datetime, timezone

router = APIRouter(prefix="/risk", tags=["risk"])


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

    class Config:
        from_attributes = True


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
