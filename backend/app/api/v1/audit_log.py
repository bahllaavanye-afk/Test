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
from pydantic import BaseModel, ConfigDict, Field, validator


router = APIRouter(prefix="/audit-log", tags=["audit-log"])


class AuditLogOut(BaseModel):
    """Schema for exposing audit log entries via the API.

    Mirrors the fields of :class:`app.models.audit_log.AuditLog` relevant for
    external consumption. Used as the response model for the ``/audit-log``
    endpoint.
    """

    id: str = Field(
        ...,
        description="Unique identifier of the audit log record.",
        example="123e4567-e89b-12d3-a456-426614174000",
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
        example="order_987",
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

    @validator("ip_address")
    def validate_ip_address(cls, v: Optional[str]) -> Optional[str]:
        """Validate that the provided IP address, if any, is syntactically correct."""
        if v is None:
            return v
        import ipaddress

        try:
            ipaddress.ip_address(v)
        except ValueError as exc:
            raise ValueError(f"Invalid IP address: {v}") from exc
        return v

    @validator("extra_data")
    def validate_extra_data(cls, v: dict) -> dict:
        """Ensure ``extra_data`` is a dictionary."""
        if not isinstance(v, dict):
            raise TypeError("extra_data must be a dict")
        return v

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
        If the user is not authenticated.
    ValueError
        If ``limit`` is ``None`` or outside the allowed range.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user not found.",
        )

    validated_limit = _validate_limit(limit)
    audit_logs = await _fetch_audit_logs(db, current_user.id, validated_limit)
    return audit_logs