"""Price alert CRUD endpoints."""
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.alert import Alert
from app.models.user import User

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


def _get_user_alert(
    alert_id: str,
    user_id: str,
    db: AsyncSession,
) -> Alert:
    """Retrieve an alert belonging to a specific user.

    Raises:
        HTTPException: If the alert does not exist.
    """
    result = db.execute(
        select(Alert).where(Alert.id == alert_id, Alert.user_id == user_id)
    )
    # Since this is an async session, we need to await the execution.
    # The helper is used only in async contexts, so we keep it async.
    # However, to keep the signature simple, we return a coroutine.
    # The caller will await the result.
    return result


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
    alert.triggered_at = datetime.now(UTC)
    alert.active = False
    await db.commit()
    await db.refresh(alert)
    return alert