"""Audit log endpoint — returns recent audit events for the current user.

Provides a read‑only API for retrieving the most recent audit log entries
associated with the authenticated user. The endpoint validates the request
parameters, enforces authentication, and returns a list of serialized audit
records.
"""

from typing import List, Optional

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
    """Schema for exposing audit log entries via the API.

    Attributes
    ----------
    id: str
        Unique identifier of the audit log record.
    action: str
        The action performed (e.g., ``login``, ``order_create``).
    resource_type: str | None
        Type of the resource affected by the action, if applicable.
    resource_id: str | None
        Identifier of the specific resource affected, if applicable.
    ip_address: str | None
        IP address from which the action originated.
    user_agent: str | None
        User‑agent string of the client that triggered the action.
    extra_data: dict
        Arbitrary additional data supplied by the audit event.
    created_at: datetime
        Timestamp when the audit record was created.
    """

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
    limit: Optional[int] = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[AuditLogOut]:
    """Return the last *limit* audit events for the authenticated user.

    Parameters
    ----------
    limit: Optional[int]
        Maximum number of records to retrieve. Must be between 1 and 500.
        Defaults to 100 if omitted.
    db: AsyncSession
        Asynchronous SQLAlchemy session injected by FastAPI dependency.
    current_user: User
        The user extracted from the authentication token.

    Returns
    -------
    List[AuditLogOut]
        A list of audit log entries ordered by creation time descending.

    Raises
    ------
    HTTPException
        If the user is not authenticated.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found.",
        )

    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == current_user.id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )
    audit_logs = result.scalars().all()
    return audit_logs