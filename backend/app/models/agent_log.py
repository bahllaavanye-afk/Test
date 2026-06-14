"""AgentActivityLog — tracks every agent and employee action in real-time."""
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AgentActivityLog(Base):
    __tablename__ = "agent_activity_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # Who did it
    employee_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # What they did
    action: Mapped[str] = mapped_column(String(256))
    tool_used: Mapped[str | None] = mapped_column(String(128))
    # Inputs/outputs (sanitized — never log secrets)
    input_summary: Mapped[str | None] = mapped_column(Text)
    output_summary: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="ok")
    error_message: Mapped[str | None] = mapped_column(Text)
    # Review
    anomaly_score: Mapped[float] = mapped_column(Float, default=0.0)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    reviewed_by: Mapped[str | None] = mapped_column(String(64))
    reviewed_at: Mapped[str | None] = mapped_column(String(64))
    review_note: Mapped[str | None] = mapped_column(Text)
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    # Context
    strategy_name: Mapped[str | None] = mapped_column(String(128))
    symbol: Mapped[str | None] = mapped_column(String(32))
    account_id: Mapped[str | None] = mapped_column(String(64))
    extra: Mapped[dict | None] = mapped_column(JSON)
