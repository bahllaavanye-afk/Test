import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(64))  # "order_submit", "order_cancel", "login", "key_add"
    resource_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    extra_data: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=lambda: {},
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __init__(
        self,
        user_id: str,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialise an AuditLog entry with defensive handling for edge cases.

        - `user_id` and `action` must be provided; a ValueError is raised otherwise.
        - Empty strings for optional fields are interpreted as None.
        - `extra_data` defaults to an empty dict if None or not a dict.
        """
        if not user_id:
            raise ValueError("user_id must be a non-empty string")
        if not action:
            raise ValueError("action must be a non-empty string")

        self.user_id = user_id
        self.action = action

        # Convert empty strings to None for optional fields
        self.resource_type = resource_type or None
        self.resource_id = resource_id or None
        self.ip_address = ip_address or None
        self.user_agent = user_agent or None

        # Ensure extra_data is a dict; protect against mutable default pitfalls
        if isinstance(extra_data, dict):
            self.extra_data = extra_data
        else:
            self.extra_data = {}

    def __repr__(self) -> str:
        return (
            f"AuditLog(id={self.id!r}, user_id={self.user_id!r}, action={self.action!r}, "
            f"resource_type={self.resource_type!r}, resource_id={self.resource_id!r}, "
            f"ip_address={self.ip_address!r}, user_agent={self.user_agent!r}, "
            f"extra_data={self.extra_data!r}, created_at={self.created_at!r})"
        )