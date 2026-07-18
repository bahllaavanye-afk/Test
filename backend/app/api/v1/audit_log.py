"""Audit log endpoint — returns recent audit events for the current user."""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.audit_log import AuditLog
from app.models.user import User
from pydantic import BaseModel, ConfigDict


router = APIRouter(prefix="/audit-log", tags=["audit-log"])


class AuditLogOut(BaseModel):
    """Schema representing an audit log entry returned to the client."""

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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[AuditLogOut]:
    """
    Retrieve the most recent audit events for the authenticated user.

    Args:
        limit: Maximum number of audit log entries to return (default 100, min 1, max 500).
        db: Asynchronous SQLAlchemy session, provided by the dependency injection system.
        current_user: The currently authenticated user, injected via dependency.

    Returns:
        A list of audit log entries ordered by creation time descending.
    """
    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == current_user.id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()