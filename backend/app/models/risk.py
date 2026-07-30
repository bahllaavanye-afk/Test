import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import String, ForeignKey, Numeric, DateTime, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


"""Risk modeling module.

Defines ORM models for risk rules and associated events used throughout the
QuantEdge platform.  The models are simple data containers with no business
logic; they are used by services that enforce risk limits, generate alerts, and
track rule violations.
"""


class RiskRule(Base):
    """ORM model representing a configurable risk rule.

    Attributes
    ----------
    id: str
        Primary key, generated as a UUID4 string.
    account_id: Optional[str]
        Identifier of the account to which the rule applies. ``None`` indicates a
        global rule that applies to all accounts.
    rule_type: str
        Category of the rule (e.g., ``"max_position"``, ``"max_drawdown"``).
    risk_bucket: Optional[str]
        Optional bucket name for grouping rules (e.g., ``"arbitrage"``, ``"ml"``).
        ``None`` denotes a global bucket.
    threshold: float
        Numeric limit that triggers the rule. Stored with high precision.
    action: str
        Action to take when the rule is breached (e.g., ``"alert"``,
        ``"halt_bucket"``, ``"halt_all"``).
    is_active: bool
        Flag indicating whether the rule is currently enforced.
    created_at: Optional[datetime]
        Timestamp when the rule was created.
    events: List[RiskEvent]
        Collection of :class:`RiskEvent` objects that have been generated for
        this rule.
    """

    __tablename__ = "risk_rules"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    account_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    rule_type: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_bucket: Mapped[Optional[str]] = mapped_column(String(16))
    threshold: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    events: Mapped[List["RiskEvent"]] = relationship(
        "RiskEvent", back_populates="rule"
    )


class RiskEvent(Base):
    """ORM model capturing an occurrence of a risk rule breach.

    Attributes
    ----------
    id: str
        Primary key, generated as a UUID4 string.
    rule_id: Optional[str]
        Foreign key referencing the associated :class:`RiskRule`. May be ``None``
        if the rule was deleted.
    account_id: str
        Identifier of the account where the event occurred.
    triggered_at: datetime
        Timestamp when the rule was triggered.
    value_at_trigger: Optional[float]
        The metric value that caused the rule to fire.
    action_taken: Optional[str]
        Description of the action performed in response to the event.
    resolved_at: Optional[datetime]
        Timestamp when the event was resolved, if applicable.
    notes: Optional[str]
        Free‑form text for additional context or comments.
    rule: Optional[RiskRule]
        Back‑reference to the parent :class:`RiskRule`.
    """

    __tablename__ = "risk_events"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    rule_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("risk_rules.id", ondelete="SET NULL")
    )
    account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.id"), index=True
    )
    triggered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    value_at_trigger: Mapped[Optional[float]] = mapped_column(Numeric(18, 6))
    action_taken: Mapped[Optional[str]] = mapped_column(String(64))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    notes: Mapped[Optional[str]] = mapped_column(Text)

    rule: Mapped[Optional["RiskRule"]] = relationship(
        "RiskRule", back_populates="events"
    )