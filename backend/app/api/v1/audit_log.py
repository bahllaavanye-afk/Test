"""Audit log endpoint — returns recent audit events for the current user."""
import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.audit_log import AuditLog
from app.models.user import User
from pydantic import BaseModel, ConfigDict

router = APIRouter(prefix="/audit-log", tags=["audit-log"])
logger = logging.getLogger(__name__)


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

    start_time = time.perf_counter()
    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == current_user.id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Structured logging of key metrics.
    logger.info(
        "audit_log.list",
        extra={
            "user_id": str(current_user.id),
            "record_count": len(rows) if rows else 0,
            "duration_ms": round(duration_ms, 2),
        },
    )

    # Ensure we always return a list, even if the query yields no rows.
    return rows if rows is not None else []