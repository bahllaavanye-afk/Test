import uuid
from datetime import datetime
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

"""SQLAlchemy models for representing risk management rules and events.

The ``RiskRule`` model defines configurable thresholds and actions that can be
applied globally or per‑account. The ``RiskEvent`` model records each occurrence
when a rule is triggered, capturing the value that caused the trigger and any
subsequent resolution information.
"""


class RiskRule(Base):
    """A configurable risk rule.

    Attributes
    ----------
    id: str
        Primary key generated as a UUID string.
    account_id: str | None
        Optional identifier of the account to which the rule applies. If ``None``,
        the rule is considered global.
    rule_type: str
        Category or name of the rule (e.g., ``\"max_drawdown\"``).
    risk_bucket: str | None
        Logical grouping for the rule (e.g., ``\"arbitrage\"`` or ``\"ml\"``). ``None``
        indicates a global bucket.
    threshold: float
        Numeric limit that, when exceeded, triggers the rule's action.
    action: str
        Action to perform when the threshold is breached (e.g., ``\"alert\"``,
        ``\"halt_bucket\"``, ``\"halt_all\"``).
    is_active: bool
        Indicates whether the rule is currently enforced.
    created_at: datetime | None
        Timestamp of rule creation.
    events: list[RiskEvent]
        Collection of :class:`RiskEvent` instances linked to this rule.
    """

    __tablename__ = "risk_rules"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    rule_type: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_bucket: Mapped[str | None] = mapped_column(String(16))  # None=global, arbitrage, ml
    threshold: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)  # alert|halt_bucket|halt_all
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    events: Mapped[list["RiskEvent"]] = relationship("RiskEvent", back_populates="rule")


class RiskEvent(Base):
    """An occurrence of a rule being triggered.

    Attributes
    ----------
    id: str
        Primary key generated as a UUID string.
    rule_id: str | None
        Foreign key to the associated :class:`RiskRule`. ``None`` if the rule was
        deleted after the event was recorded.
    account_id: str
        Identifier of the account where the event occurred.
    triggered_at: datetime
        Timestamp when the rule was triggered.
    value_at_trigger: float | None
        The metric value that caused the trigger.
    action_taken: str | None
        Description of the action performed in response to the trigger.
    resolved_at: datetime | None
        Timestamp when the event was resolved, if applicable.
    notes: str | None
        Additional free‑form information about the event.
    rule: RiskRule | None
        ORM relationship back to the associated rule.
    """

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