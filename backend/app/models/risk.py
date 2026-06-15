import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class RiskRule(Base):
    __tablename__ = "risk_rules"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str | None] = mapped_column(String, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=True, index=True)
    rule_type: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_bucket: Mapped[str | None] = mapped_column(String(16))  # None=global, arbitrage, ml
    threshold: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)  # alert|halt_bucket|halt_all
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    events: Mapped[list["RiskEvent"]] = relationship("RiskEvent", back_populates="rule")


class RiskEvent(Base):
    __tablename__ = "risk_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    rule_id: Mapped[str | None] = mapped_column(String, ForeignKey("risk_rules.id", ondelete="SET NULL"))
    account_id: Mapped[str] = mapped_column(String, ForeignKey("accounts.id"), index=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value_at_trigger: Mapped[float | None] = mapped_column(Numeric(18, 6))
    action_taken: Mapped[str | None] = mapped_column(String(64))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)

    rule: Mapped["RiskRule | None"] = relationship("RiskRule", back_populates="events")
