import uuid
from datetime import datetime, timezone
from sqlalchemy import String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


"""SQLAlchemy ORM model for audit log entries.

This module defines the :class:`AuditLog` model which records user actions,
including metadata such as IP address, user agent, and any additional data
relevant to the event. The model is used throughout the platform for
compliance, debugging, and operational monitoring.
"""


class AuditLog(Base):
    """Represent a single audit log entry.

    Attributes
    ----------
    id : Mapped[str]
        Primary key, generated as a UUID string.
    user_id : Mapped[str]
        Identifier of the user who performed the action; foreign key to
        ``users.id``.
    action : Mapped[str]
        Type of action performed (e.g., ``order_submit``, ``order_cancel``,
        ``login``, ``key_add``).
    resource_type : Mapped[str | None]
        Optional type of the resource the action targets (e.g., ``order``,
        ``account``).
    resource_id : Mapped[str | None]
        Optional identifier of the specific resource instance.
    ip_address : Mapped[str | None]
        IP address from which the action originated.
    user_agent : Mapped[str | None]
        User‑agent string of the client making the request.
    extra_data : Mapped[dict]
        Arbitrary JSON‑serialisable data providing additional context.
    created_at : Mapped[datetime]
        Timestamp of when the log entry was created, stored in UTC.
    """

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(64))  # "order_submit", "order_cancel", "login", "key_add"
    resource_type: Mapped[str | None] = mapped_column(String(32))
    resource_id: Mapped[str | None] = mapped_column(String(64))
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(256))
    extra_data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )