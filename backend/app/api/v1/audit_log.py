"""Audit log endpoint — returns recent audit events for the current user."""
from datetime import datetime
from typing import Any, Dict, Optional

import ipaddress
from fastapi import APIRouter, Depends, Query, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.audit_log import AuditLog
from app.models.user import User

router = APIRouter(prefix="/audit-log", tags=["audit-log"])


class AuditLogOut(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier of the audit log entry.",
        example="a1b2c3d4e5f6g7h8i9j0",
    )
    action: str = Field(
        ...,
        description="Action performed that generated the audit event.",
        example="login",
    )
    resource_type: Optional[str] = Field(
        None,
        description="Type of the resource the action was performed on, if applicable.",
        example="order",
    )
    resource_id: Optional[str] = Field(
        None,
        description="Identifier of the specific resource, if applicable.",
        example="ORD-12345",
    )
    ip_address: Optional[str] = Field(
        None,
        description="IP address of the client that triggered the event.",
        example="192.168.1.100",
    )
    user_agent: Optional[str] = Field(
        None,
        description="User‑Agent string from the client request.",
        example="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    )
    extra_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional information attached to the event.",
        example={"detail": "Two‑factor authentication succeeded"},
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the audit event was recorded (UTC).",
        example="2023-07-21T14:30:00Z",
    )

    model_config = ConfigDict(from_attributes=True)

    @validator("ip_address")
    def validate_ip(cls, v: Optional[str]) -> Optional[str]:
        """Validate that the IP address, if provided, is syntactically correct."""
        if v is None:
            return v
        try:
            ipaddress.ip_address(v)
        except ValueError as exc:
            raise ValueError(f"Invalid IP address: {v}") from exc
        return v

    @validator("extra_data", pre=True, always=True)
    def ensure_extra_data_dict(cls, v: Any) -> Dict[str, Any]:
        """Guarantee that extra_data is a dictionary."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        raise ValueError("extra_data must be a dictionary")


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

    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == current_user.id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return rows