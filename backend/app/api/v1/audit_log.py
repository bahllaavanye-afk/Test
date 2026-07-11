"""Audit log endpoint — returns recent audit events for the current user."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import time
import asyncio

from app.database import get_db
from app.api.deps import get_current_user
from app.models.audit_log import AuditLog
from app.models.user import User
from pydantic import BaseModel, ConfigDict

router = APIRouter(prefix="/audit-log", tags=["audit-log"])

# Simple in‑process cache for recent audit logs.
# Structure: {user_id: (timestamp, [AuditLogOut, ...])}
_AUDIT_LOG_CACHE: dict[int, tuple[float, list["AuditLogOut"]]] = {}
_CACHE_TTL_SECONDS = 30  # refresh cache after this many seconds.
_CACHE_LOCK = asyncio.Lock()


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


async def _get_cached_audit_logs(
    user_id: int, limit: int, db: AsyncSession
) -> list[AuditLogOut]:
    """Return cached audit logs if fresh, otherwise query the DB and update cache."""
    now = time.time()
    async with _CACHE_LOCK:
        cached = _AUDIT_LOG_CACHE.get(user_id)
        if cached:
            ts, data = cached
            if now - ts < _CACHE_TTL_SECONDS and len(data) >= limit:
                # Cache is fresh and contains at least the requested number of entries.
                return data[:limit]

    # Cache miss or stale – fetch from DB.
    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == user_id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )
    logs = result.scalars().all()

    # Convert to output models.
    output = [AuditLogOut.from_orm(log) for log in logs]

    # Update cache.
    async with _CACHE_LOCK:
        _AUDIT_LOG_CACHE[user_id] = (now, output)

    return output


@router.get("/", response_model=list[AuditLogOut])
async def list_audit_log(
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the last N audit events for the authenticated user."""
    return await _get_cached_audit_logs(current_user.id, limit, db)