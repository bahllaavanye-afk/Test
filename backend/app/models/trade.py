import uuid
from datetime import datetime
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Integer, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base

"""Models for the trading core.

This module defines the ORM models used by the QuantEdge trading platform.
"""

class Trade(Base):
    """SQLAlchemy model representing a trade execution.

    Attributes
    ----------
    id : str
        Unique identifier for the trade.
    account_id : str
        Foreign key referencing the account that placed the trade.
    strategy_id : Optional[str]
        Foreign key referencing the strategy that generated the trade; may be null.
    strategy_name : Optional[str]
        Denormalized strategy name for fast attribution queries.
    symbol : str
        Ticker symbol of the traded instrument.
    side : str
        Trade side, typically ``'buy'`` or ``'sell'``.
    entry_price : float
        Execution price when the position was opened.
    exit_price : float
        Execution price when the position was closed.
    quantity : float
        Number of units traded.
    realized_pnl : float
        Realized profit or loss of the trade.
    fees : float
        Total fees charged for the trade.
    opened_at : datetime
        Timestamp when the trade was opened.
    closed_at : datetime
        Timestamp when the trade was closed.
    hold_seconds : Optional[int]
        Duration of the trade in seconds.
    raw_payload : dict
        Original payload received from the execution source.
    """

    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_account_closed", "account_id", "closed_at"),
        Index("ix_trades_strategy_closed", "strategy_id", "closed_at"),
    )

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        doc="Unique identifier for the trade."
    )
    account_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("accounts.id", ondelete="CASCADE"),
        index=True,
        doc="Foreign key referencing the account that placed the trade."
    )
    strategy_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("strategies.id", ondelete="SET NULL"),
        index=True,
        doc="Foreign key referencing the strategy that generated the trade; may be null."
    )
    strategy_name: Mapped[str | None] = mapped_column(
        String(128),
        index=True,
        doc="Denormalized strategy name for fast attribution queries."
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Ticker symbol of the traded instrument."
    )
    side: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        doc="Trade side, typically 'buy' or 'sell'."
    )
    entry_price: Mapped[float] = mapped_column(
        Numeric(18, 8),
        nullable=False,
        doc="Execution price when the position was opened."
    )
    exit_price: Mapped[float] = mapped_column(
        Numeric(18, 8),
        nullable=False,
        doc="Execution price when the position was closed."
    )
    quantity: Mapped[float] = mapped_column(
        Numeric(18, 8),
        nullable=False,
        doc="Number of units traded."
    )
    realized_pnl: Mapped[float] = mapped_column(
        Numeric(18, 8),
        nullable=False,
        doc="Realized profit or loss of the trade."
    )
    fees: Mapped[float] = mapped_column(
        Numeric(18, 8),
        default=0,
        doc="Total fees charged for the trade."
    )
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="Timestamp when the trade was opened."
    )
    closed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Timestamp when the trade was closed."
    )
    hold_seconds: Mapped[int | None] = mapped_column(
        Integer,
        doc="Duration of the trade in seconds."
    )
    raw_payload: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        doc="Original payload received from the execution source."
    )