"""Audit log endpoint — returns recent audit events for the current user.

Provides a read‑only API for retrieving the most recent audit log entries
associated with the authenticated user. The endpoint validates the request
parameters, enforces authentication, and returns a list of serialized audit
records.

The module defines the FastAPI router, the response schema, and the handler
function.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.audit_log import AuditLog
from app.models.user import User
from pydantic import BaseModel, ConfigDict

# Constants
DEFAULT_LIMIT = 100
MIN_LIMIT = 1
MAX_LIMIT = 500
AUTH_ERROR_STATUS = status.HTTP_401_UNAUTHORIZED
AUTH_ERROR_DETAIL = "Authenticated user not found."
BAD_REQUEST_STATUS = status.HTTP_400_BAD_REQUEST
BAD_REQUEST_DETAIL = "Invalid request parameters."
ROUTER_PREFIX = "/audit-log"
ROUTER_TAGS = ["audit-log"]

router = APIRouter(prefix=ROUTER_PREFIX, tags=ROUTER_TAGS)


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
    resource_type: Optional[str]
    resource_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    extra_data: dict
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


def _validate_limit(limit: Optional[int]) -> int:
    """Validate the ``limit`` query parameter.

    Handles ``None`` and out‑of‑range values, falling back to ``DEFAULT_LIMIT`` for
    ``None`` and raising ``ValueError`` for other invalid inputs.

    Parameters
    ----------
    limit: Optional[int]
        The raw limit value from the request.

    Returns
    -------
    int
        The validated limit.

    Raises
    ------
    ValueError
        If ``limit`` is out of the allowed range.
    """
    if limit is None:
        return DEFAULT_LIMIT
    if not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if limit < MIN_LIMIT or limit > MAX_LIMIT:
        raise ValueError(
            f"limit must be between {MIN_LIMIT} and {MAX_LIMIT}, got {limit}"
        )
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
    if db is None:
        return []
    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == user_id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )
    # ``scalars().all()`` returns an empty list when no rows are found,
    # but guard against a ``None`` result for extra safety.
    logs = result.scalars().all()
    return logs if logs is not None else []


@router.get("/", response_model=List[AuditLogOut])
async def list_audit_log(
    limit: Optional[int] = Query(default=DEFAULT_LIMIT, ge=MIN_LIMIT, le=MAX_LIMIT),
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
        If the user is not authenticated or request parameters are invalid.
    """
    if current_user is None:
        raise HTTPException(status_code=AUTH_ERROR_STATUS, detail=AUTH_ERROR_DETAIL)

    if getattr(current_user, "id", None) is None:
        raise HTTPException(status_code=BAD_REQUEST_STATUS, detail=BAD_REQUEST_DETAIL)

    try:
        validated_limit = _validate_limit(limit)
    except ValueError as exc:
        raise HTTPException(status_code=BAD_REQUEST_STATUS, detail=str(exc))

    audit_logs = await _fetch_audit_logs(db, current_user.id, validated_limit)
    # Ensure the response is always a list, even if the query returned None.
    return audit_logs if audit_logs is not None else []