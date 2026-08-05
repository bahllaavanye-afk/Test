import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class AuditLog(Base):
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


class AuditLogCreate(BaseModel):
    user_id: str = Field(..., description="ID of the user performing the action")
    action: str = Field(..., description="Action performed (e.g., order_submit, order_cancel, login, key_add)", examples=["order_submit", "order_cancel"])
    resource_type: str | None = Field(None, description="Type of resource affected (e.g., order, user, key)", examples=["order", "user"])
    resource_id: str | None = Field(None, description="ID of the resource affected", examples=["ord_12345", "usr_67890"])
    ip_address: str | None = Field(None, description="IP address of the requester", examples=["192.168.1.1", "10.0.0.1"])
    user_agent: str | None = Field(None, description="User agent string of the requester", examples=["Mozilla/5.0 ..."])
    extra_data: dict = Field(default_factory=dict, description="Additional metadata as a JSON object", examples=[{"reason": "insufficient funds"}])

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed_actions = {"order_submit", "order_cancel", "login", "key_add"}
        if v not in allowed_actions:
            raise ValueError(f"action must be one of {allowed_actions}")
        return v

    @field_validator("ip_address")
    @classmethod
    def validate_ip_address(cls, v: str | None) -> str | None:
        if v is not None:
            import ipaddress
            try:
                ipaddress.ip_address(v)
            except ValueError:
                raise ValueError("ip_address must be a valid IP address")
        return v


class AuditLogResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the audit log entry")
    user_id: str = Field(..., description="ID of the user who performed the action")
    action: str = Field(..., description="Action performed")
    resource_type: str | None = Field(None, description="Type of resource affected")
    resource_id: str | None = Field(None, description="ID of the resource affected")
    ip_address: str | None = Field(None, description="IP address of the requester")
    user_agent: str | None = Field(None, description="User agent string of the requester")
    extra_data: dict = Field(default_factory=dict, description="Additional metadata as a JSON object")
    created_at: datetime = Field(..., description="Timestamp when the audit log entry was created")