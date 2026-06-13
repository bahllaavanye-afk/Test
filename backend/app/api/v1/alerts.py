"""Price alert CRUD endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from app.database import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.alert import Alert

router = APIRouter(prefix="/alerts", tags=["alerts"])

AlertType = Literal["price_above", "price_below", "change_pct", "rsi_overbought", "rsi_oversold"]


class AlertCreate(BaseModel):
    model_config = ConfigDict(strict=True)
    symbol: str
    type: AlertType
    value: float
    note: str = Field(default="", max_length=512)


class AlertResponse(BaseModel):
    id: str
    symbol: str
    type: str
    value: float
    note: str
    active: bool
    triggered_at: datetime | None
    created_at: datetime


@router.get("/", response_model=list[AlertResponse])
async def list_alerts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Alert)
        .where(Alert.user_id == str(current_user.id))
        .order_by(Alert.created_at.desc())
        .limit(100)
    )
    return result.scalars().all()


@router.post("/", response_model=AlertResponse, status_code=201)
async def create_alert(
    body: AlertCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    alert = Alert(
        user_id=str(current_user.id),
        symbol=body.symbol.upper(),
        type=body.type,
        value=body.value,
        note=body.note,
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return alert


@router.delete("/{alert_id}", status_code=204)
async def delete_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Alert).where(Alert.id == alert_id, Alert.user_id == str(current_user.id))
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    await db.delete(alert)
    await db.commit()


@router.patch("/{alert_id}/trigger", response_model=AlertResponse)
async def trigger_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mark an alert as triggered (called by the price monitoring task)."""
    result = await db.execute(
        select(Alert).where(Alert.id == alert_id, Alert.user_id == str(current_user.id))
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.triggered_at = datetime.now(timezone.utc)
    alert.active = False
    await db.commit()
    await db.refresh(alert)
    return alert
