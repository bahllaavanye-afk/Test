"""Audit log endpoint — returns recent audit events for the current user."""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.api.deps import get_current_user
from app.models.audit_log import AuditLog
from app.models.user import User
from pydantic import BaseModel, ConfigDict


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


@router.get("/", response_model=List[AuditLogOut])
async def list_audit_log(
    limit: int = Query(default=100, ge=1, le=500),
    action: Optional[str] = Query(
        default=None,
        description="Filter logs by action type (e.g., 'login', 'order_created').",
    ),
    since: Optional[datetime] = Query(
        default=None,
        description="Return logs created after this timestamp (UTC).",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[AuditLogOut]:
    """Return the most recent audit events for the authenticated user.

    The endpoint supports optional filtering:
    - `action`: restrict results to a specific audit action.
    - `since`: only include events created after the supplied timestamp.

    Defensive handling ensures the `limit` respects the allowed range.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found.",
        )

    # Build the base WHERE clause for the current user.
    conditions = [AuditLog.user_id == current_user.id]

    # Optional action filter.
    if action:
        conditions.append(AuditLog.action == action)

    # Optional time filter – ensure the timestamp is not in the future.
    if since:
        now = datetime.utcnow()
        if since > now:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="`since` parameter cannot be in the future.",
            )
        conditions.append(AuditLog.created_at >= since)

    query = (
        select(AuditLog)
        .where(and_(*conditions))
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )

    result = await db.execute(query)
    rows = result.scalars().all()
    return rows