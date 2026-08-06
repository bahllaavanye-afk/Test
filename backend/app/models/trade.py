import uuid
import logging
from datetime import datetime
from sqlalchemy import (
    String,
    ForeignKey,
    Numeric,
    DateTime,
    Integer,
    JSON,
    Index,
    event,
)
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base

logger = logging.getLogger(__name__)

class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_account_closed", "account_id", "closed_at"),
        Index("ix_trades_strategy_closed", "strategy_id", "closed_at"),
    )

    # Class‑level counter to track number of trade signals processed
    _signal_counter: int = 0

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    strategy_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("strategies.id", ondelete="SET NULL"), index=True
    )
    # Denormalized for fast attribution queries (avoids JOIN to strategies table)
    strategy_name: Mapped[str | None] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    exit_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    fees: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    hold_seconds: Mapped[int | None] = mapped_column(Integer)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)

# ----------------------------------------------------------------------
# Monitoring hooks – structured logging of key trade metrics
# ----------------------------------------------------------------------
@event.listens_for(Trade, "before_insert")
def _increment_signal_counter(mapper, connection, target):
    """Increment the class‑level signal counter before a trade row is persisted."""
    Trade._signal_counter += 1
    # Store the counter on the instance for later logging
    target._signal_seq = Trade._signal_counter

@event.listens_for(Trade, "after_insert")
def _log_trade_metrics(mapper, connection, target):
    """Log essential trade metrics at INFO level after the row is inserted."""
    try:
        execution_seconds = (
            (target.closed_at - target.opened_at).total_seconds()
            if target.closed_at and target.opened_at
            else None
        )
        logger.info(
            "Trade recorded",
            trade_id=target.id,
            signal_seq=getattr(target, "_signal_seq", None),
            symbol=target.symbol,
            side=target.side,
            quantity=float(target.quantity),
            pnl=float(target.realized_pnl),
            execution_time_seconds=execution_seconds,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to log trade metrics: %s", exc)