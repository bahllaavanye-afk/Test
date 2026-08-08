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
from pydantic import BaseModel, ConfigDict, Field, validator

from app.api.deps import get_current_user
from app.database import get_db
from app.models.audit_log import AuditLog
from app.models.user import User

# Constants
DEFAULT_LIMIT = 100
MIN_LIMIT = 1
MAX_LIMIT = 500
AUTH_ERROR_STATUS = status.HTTP_401_UNAUTHORIZED
AUTH_ERROR_DETAIL = "Authenticated user not found."
ROUTER_PREFIX = "/audit-log"
ROUTER_TAGS = ["audit-log"]

router = APIRouter(prefix=ROUTER_PREFIX, tags=ROUTER_TAGS)


class AuditLogOut(BaseModel):
    """Schema for exposing audit log entries via the API.

    Mirrors the fields of :class:`app.models.audit_log.AuditLog` that are
    relevant for external consumption.
    """

    id: str = Field(
        ...,
        description="Unique identifier of the audit log record.",
        example="a1b2c3d4e5f6g7h8i9j0",
    )
    action: str = Field(
        ...,
        description="The action performed (e.g., ``login``, ``order_create``).",
        example="login",
    )
    resource_type: Optional[str] = Field(
        None,
        description="Type of the resource affected by the action, if applicable.",
        example="order",
    )
    resource_id: Optional[str] = Field(
        None,
        description="Identifier of the specific resource affected, if applicable.",
        example="order-12345",
    )
    ip_address: Optional[str] = Field(
        None,
        description="IP address from which the action originated.",
        example="192.168.1.1",
    )
    user_agent: Optional[str] = Field(
        None,
        description="User‑agent string of the client that triggered the action.",
        example="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    )
    extra_data: dict = Field(
        default_factory=dict,
        description="Arbitrary additional data supplied by the audit event.",
        example={"key": "value"},
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the audit record was created.",
        example="2023-01-01T12:00:00Z",
    )

    model_config = ConfigDict(from_attributes=True)

    @validator("extra_data", pre=True, always=True)
    def ensure_extra_data_is_dict(cls, v):
        """Guarantee that ``extra_data`` is always a dictionary."""
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("extra_data must be a dictionary")
        return v

    @validator("ip_address")
    def validate_ip_address(cls, v):
        """Very light validation of IPv4/IPv6 address format."""
        if v is None:
            return v
        parts = v.split(".")
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            return v
        # Basic IPv6 check (contains ':')
        if ":" in v:
            return v
        raise ValueError("ip_address must be a valid IPv4 or IPv6 address")


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
    if limit < MIN_LIMIT or limit > MAX_LIMIT:
        raise ValueError(f"limit must be between {MIN_LIMIT} and {MAX_LIMIT}, got {limit}")
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
        If the user is not authenticated.
    ValueError
        If ``limit`` is ``None`` or outside the allowed range.
    """
    if current_user is None:
        raise HTTPException(
            status_code=AUTH_ERROR_STATUS,
            detail=AUTH_ERROR_DETAIL,
        )

    validated_limit = _validate_limit(limit)
    audit_logs = await _fetch_audit_logs(db, current_user.id, validated_limit)
    return audit_logs