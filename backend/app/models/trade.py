import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Union

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


"""SQLAlchemy ORM models for trade data.

This module defines the ``Trade`` model, which stores information about
individual trades executed by the platform. The model includes fields for
identifiers, pricing, quantities, timestamps, and raw payload data.
"""


class Trade(Base):
    """ORM model representing a single trade execution.

    Attributes
    ----------
    id : Mapped[str]
        Primary key for the trade record. Generated as a UUID string.
    account_id : Mapped[str]
        Foreign key referencing the account that placed the trade.
    strategy_id : Mapped[Optional[str]]
        Foreign key referencing the strategy that generated the trade.
    strategy_name : Mapped[Optional[str]]
        Denormalized strategy name for fast attribution queries.
    symbol : Mapped[str]
        Trading symbol (e.g., ticker) for the trade.
    side : Mapped[str]
        Trade side, typically ``'buy'`` or ``'sell'``.
    entry_price : Mapped[float]
        Execution price when the position was opened.
    exit_price : Mapped[float]
        Execution price when the position was closed.
    quantity : Mapped[float]
        Number of units/contracts traded.
    realized_pnl : Mapped[float]
        Realized profit and loss from the trade.
    fees : Mapped[float]
        Total fees incurred for the trade.
    opened_at : Mapped[datetime]
        Timestamp when the trade was opened (timezone‑aware).
    closed_at : Mapped[datetime]
        Timestamp when the trade was closed (timezone‑aware).
    hold_seconds : Mapped[Optional[int]]
        Duration the trade was held, in seconds.
    raw_payload : Mapped[Dict[str, Any]]
        Original payload received from the execution source.
    """

    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_account_closed", "account_id", "closed_at"),
        Index("ix_trades_strategy_closed", "strategy_id", "closed_at"),
    )

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    account_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("accounts.id", ondelete="CASCADE"),
        index=True,
    )
    strategy_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("strategies.id", ondelete="SET NULL"),
        index=True,
    )
    # Denormalized for fast attribution queries (avoids JOIN to strategies table)
    strategy_name: Mapped[Optional[str]] = mapped_column(
        String(128), index=True
    )
    symbol: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    side: Mapped[str] = mapped_column(
        String(8), nullable=False
    )
    entry_price: Mapped[float] = mapped_column(
        Numeric(18, 8), nullable=False
    )
    exit_price: Mapped[float] = mapped_column(
        Numeric(18, 8), nullable=False
    )
    quantity: Mapped[float] = mapped_column(
        Numeric(18, 8), nullable=False
    )
    realized_pnl: Mapped[float] = mapped_column(
        Numeric(18, 8), nullable=False
    )
    fees: Mapped[float] = mapped_column(
        Numeric(18, 8), default=0
    )
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    closed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    hold_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    raw_payload: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict
    )