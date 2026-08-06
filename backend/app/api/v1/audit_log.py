"""Audit log endpoint — returns recent audit events for the current user.

Provides a read‑only API for retrieving the most recent audit log entries
associated with the authenticated user. The endpoint validates the request
parameters, enforces authentication, and returns a list of serialized audit
records.
"""

from typing import List, Optional

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


def _validate_user(current_user: Optional[User]) -> User:
    """Ensure an authenticated user is present.

    Raises
    ------
    HTTPException
        If ``current_user`` is ``None``.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found.",
        )
    return current_user


def _validate_limit(limit: Optional[int]) -> int:
    """Validate the ``limit`` query parameter.

    Parameters
    ----------
    limit: Optional[int]
        The requested number of records.

    Returns
    -------
    int
        The validated limit.

    Raises
    ------
    ValueError
        If ``limit`` is ``None`` or outside the allowed range (1‑500).
    """
    if limit is None:
        raise ValueError("limit must not be None")
    if limit < 1 or limit > 500:
        raise ValueError(f"limit must be between 1 and 500, got {limit}")
    return limit


async def _fetch_audit_logs(
    db: AsyncSession, user_id: int, limit: int
) -> List[AuditLog]:
    """Retrieve the most recent audit log entries for a user.

    Parameters
    ----------
    db: AsyncSession
        The asynchronous SQLAlchemy session.
    user_id: int
        Identifier of the user whose logs are being fetched.
    limit: int
        Maximum number of records to return.

    Returns
    -------
    List[AuditLog]
        AuditLog objects ordered by ``created_at`` descending.
    """
    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == user_id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


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
    ValueError
        If ``limit`` is ``None`` or outside the allowed range.
    """
    user = _validate_user(current_user)
    validated_limit = _validate_limit(limit)
    return await _fetch_audit_logs(db, user.id, validated_limit)