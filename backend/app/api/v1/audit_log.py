"""Audit log endpoint — returns recent audit events for the current user."""
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.audit_log import AuditLog
from app.models.user import User
from pydantic import BaseModel, ConfigDict
from datetime import datetime

router = APIRouter(prefix="/audit-log", tags=["audit-log"])


class AuditLogOut(BaseModel):
    id: str
    action: str
    resource_type: str | None
    resource_id: str | None
    ip_address: str | None
    user_agent: str | None
    extra_data: dict
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


@router.get("/", response_model=list[AuditLogOut])
async def list_audit_log(
    limit: int | None = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[AuditLogOut]:
    """Return the last N audit events for the authenticated user.

    Handles edge cases:
    - `limit` being None.
    - `limit` outside allowed bounds (defensive clamping).
    - Missing or empty result set.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found.",
        )

    # Defensive handling for limit being None or out of range.
    if limit is None:
        limit = 100
    elif limit < 1:
        limit = 1
    elif limit > 500:
        limit = 500

    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == current_user.id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return rows