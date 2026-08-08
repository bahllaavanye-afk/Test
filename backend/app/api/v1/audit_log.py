"""Audit log endpoint — returns recent audit events for the current user.

Provides a read‑only API for retrieving the most recent audit log entries
associated with the authenticated user. The endpoint validates the request
parameters, enforces authentication, and returns a list of serialized audit
records.

The module defines the FastAPI router, the response schema, and the handler
function.
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.audit_log import AuditLog
from app.models.user import User
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit-log", tags=["audit-log"])


class AuditLogOut(BaseModel):
    """Schema for exposing audit log entries via the API.

    This model mirrors the fields of :class:`app.models.audit_log.AuditLog` that
    are relevant for external consumption. It is used as the response model for
    the ``/audit-log`` endpoint.

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


def _validate_limit(limit: Optional[int]) -> int:
    """Validate the ``limit`` query parameter.

    Ensures the limit is not ``None`` and lies within the allowed range.
    Raises ``ValueError`` if validation fails.

    Parameters
    ----------
    limit: Optional[int]
        The raw limit value from the request.

    Returns
    -------
    int
        The validated limit.
    """
    if limit is None:
        raise ValueError("limit must not be None")
    if limit < 1 or limit > 500:
        raise ValueError(f"limit must be between 1 and 500, got {limit}")
    return limit


async def _fetch_audit_logs(
    db: AsyncSession, user_id: str, limit: int
) -> List[AuditLog]:
    """Fetch the most recent audit logs for a given user.

    Parameters
    ----------
    db: AsyncSession
        Asynchronous SQLAlchemy session.
    user_id: str
        Identifier of the user whose audit logs are being retrieved.
    limit: int
        Maximum number of records to return.

    Returns
    -------
    List[AuditLog]
        List of audit log entries ordered by creation time descending.
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
    """Return the most recent audit events for the authenticated user.

    The endpoint fetches up to ``limit`` audit log records belonging to the
    current user, ordered by creation time in descending order.

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
        If the user is not authenticated or if an error occurs during processing.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found.",
        )

    try:
        validated_limit = _validate_limit(limit)
    except ValueError as ve:
        logger.error(
            "Invalid limit parameter",
            extra={"limit": limit, "user_id": current_user.id, "error": str(ve)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        ) from ve

    try:
        audit_logs = await _fetch_audit_logs(db, current_user.id, validated_limit)
    except SQLAlchemyError as db_err:
        logger.error(
            "Database error while fetching audit logs",
            extra={"user_id": current_user.id, "limit": validated_limit, "error": str(db_err)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit logs due to a server error.",
        ) from db_err
    except Exception as exc:
        logger.error(
            "Unexpected error while fetching audit logs",
            extra={"user_id": current_user.id, "limit": validated_limit, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
        ) from exc

    return audit_logs