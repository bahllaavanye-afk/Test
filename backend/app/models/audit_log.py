import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from app.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.id"),
        index=True,
    )
    action: Mapped[str] = mapped_column(String(64))  # "order_submit", "order_cancel", "login", "key_add"
    resource_type: Mapped[Optional[str]] = mapped_column(String(32))
    resource_id: Mapped[Optional[str]] = mapped_column(String(64))
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(String(256))
    extra_data: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    def __init__(
        self,
        *,
        user_id: str,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
    ) -> None:
        # Basic non‑None enforcement for required fields
        if not user_id:
            raise ValueError("user_id must be provided")
        if not action:
            raise ValueError("action must be provided")

        self.user_id = user_id
        self.action = action
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.ip_address = ip_address
        self.user_agent = user_agent
        # Convert None or empty collections to a safe default dict
        self.extra_data = extra_data if isinstance(extra_data, dict) else {}
        self.created_at = created_at or datetime.now(timezone.utc)

    @validates("action")
    def _validate_action(self, key: str, value: str) -> str:
        # Ensure a non‑empty string and truncate if it exceeds column limit
        if not value:
            raise ValueError("action cannot be empty")
        return value[:64] if len(value) > 64 else value

    @validates("resource_type")
    def _validate_resource_type(self, key: str, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value[:32] if len(value) > 32 else value

    @validates("resource_id")
    def _validate_resource_id(self, key: str, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value[:64] if len(value) > 64 else value

    @validates("ip_address")
    def _validate_ip_address(self, key: str, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value[:45] if len(value) > 45 else value

    @validates("user_agent")
    def _validate_user_agent(self, key: str, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value[:256] if len(value) > 256 else value

    @validates("extra_data")
    def _validate_extra_data(self, key: str, value: Any) -> Dict[str, Any]:
        # Accept only dict‑like structures; coerce None or empty collections to empty dict
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise ValueError("extra_data must be a dictionary")

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id!r} user_id={self.user_id!r} action={self.action!r} "
            f"resource_type={self.resource_type!r} resource_id={self.resource_id!r}>"
        )