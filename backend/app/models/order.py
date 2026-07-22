import uuid
import logging
from datetime import datetime, date
from typing import Any, Dict, Optional

from sqlalchemy import (
    String,
    ForeignKey,
    Numeric,
    DateTime,
    Date,
    Integer,
    JSON,
    Index,
    event,
    select,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, Session

from app.database import Base
from app.models.base import TimestampMixin

logger = logging.getLogger(__name__)


class Order(Base, TimestampMixin):
    """
    ORM model representing an order placed by an account.

    The model now includes helper methods and SQLAlchemy event listeners
    that emit structured INFO‑level logs containing key operational metrics:
    * ``signal_count`` – total number of orders for the associated strategy.
    * ``execution_time`` – seconds elapsed between submission and fill.
    * ``pnl`` – profit & loss calculated from filled quantity and average fill price.
    """

    __tablename__ = "orders"
    __table_args__ = (
        # Composite indexes for the most common query patterns
        Index("ix_orders_account_status", "account_id", "status"),
        Index("ix_orders_account_created", "account_id", "created_at"),
        Index("ix_orders_symbol_status", "symbol", "status"),
    )

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.id", ondelete="CASCADE")
    )
    strategy_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("strategies.id", ondelete="SET NULL")
    )
    broker_order_id: Mapped[Optional[str]] = mapped_column(
        String(128), index=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # buy|sell
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)  # market|limit|stop
    quantity: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    limit_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    stop_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )
    filled_qty: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    avg_fill_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    time_in_force: Mapped[str] = mapped_column(String(8), default="GTC")
    execution_algo: Mapped[Optional[str]] = mapped_column(String(32))  # market|limit_first|twap|vwap
    submitted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    filled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    raw_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Bracket / advanced order fields
    take_profit_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    stop_loss_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    trailing_stop_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))  # e.g. 2.0 = 2%
    notional: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))  # buy $500 worth
    bracket_parent_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("orders.id", ondelete="SET NULL")
    )
    risk_reward_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))

    # Cross-desk tracking — one order shape for every desk (equity/crypto/option/...).
    asset_class: Mapped[str] = mapped_column(
        String(16), nullable=False, default="equity"
    )
    underlying_symbol: Mapped[Optional[str]] = mapped_column(String(32))  # options: the underlying
    expiry: Mapped[Optional[date]] = mapped_column(Date)  # options/futures
    strike: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))  # options
    option_right: Mapped[Optional[str]] = mapped_column(String(4))  # call|put
    contract_multiplier: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )

    account: Mapped["Account"] = relationship(
        "Account", back_populates="orders"
    )
    fills: Mapped[list["Fill"]] = relationship(
        "Fill", back_populates="order"
    )
    slippage: Mapped[list["SlippageRecord"]] = relationship(
        "SlippageRecord", back_populates="order"
    )

    # --------------------------------------------------------------------- #
    # Helper methods for monitoring
    # --------------------------------------------------------------------- #

    def execution_time_seconds(self) -> Optional[float]:
        """
        Returns the elapsed time in seconds between order submission and fill.
        Returns ``None`` if either timestamp is missing.
        """
        if self.submitted_at and self.filled_at:
            return (self.filled_at - self.submitted_at).total_seconds()
        return None

    def pnl(self) -> Optional[float]:
        """
        Calculates a simple profit‑and‑loss figure for the order.

        For *buy* orders the P&L is considered as ``filled_qty * avg_fill_price``.
        For *sell* orders it is the negative of that amount.  Returns ``None``
        if the required fields are not populated.
        """
        if self.filled_qty and self.avg_fill_price:
            base = float(self.filled_qty) * float(self.avg_fill_price)
            return base if self.side.lower() == "sell" else -base
        return None

    def signal_count(self, session: Session) -> int:
        """
        Retrieves the total number of orders associated with the same strategy.
        """
        if not self.strategy_id:
            return 0
        stmt = select(func.count()).select_from(Order).where(
            Order.strategy_id == self.strategy_id
        )
        return session.execute(stmt).scalar_one()


# --------------------------------------------------------------------- #
# SQLAlchemy event listeners for structured logging
# --------------------------------------------------------------------- #

@event.listens_for(Order, "after_insert")
def _log_order_created(mapper, connection, target: Order) -> None:
    """
    Logs creation of a new order with its identifier and strategy context.
    """
    logger.info(
        "order_created",
        extra={
            "order_id": target.id,
            "strategy_id": target.strategy_id,
            "symbol": target.symbol,
            "side": target.side,
            "order_type": target.order_type,
            "quantity": float(target.quantity) if target.quantity else None,
        },
    )


@event.listens_for(Order, "after_update")
def _log_order_metrics(mapper, connection, target: Order) -> None:
    """
    When an order transitions to a filled state, emit execution metrics.
    """
    # Detect transition to filled status
    if target.status.lower() == "filled":
        exec_time = target.execution_time_seconds()
        pnl_value = target.pnl()

        # Use a temporary Session bound to the same connection to fetch signal count
        session = Session(bind=connection)
        try:
            signal_cnt = target.signal_count(session)
        finally:
            session.close()

        logger.info(
            "order_filled",
            extra={
                "order_id": target.id,
                "strategy_id": target.strategy_id,
                "symbol": target.symbol,
                "filled_qty": float(target.filled_qty),
                "avg_fill_price": float(target.avg_fill_price)
                if target.avg_fill_price
                else None,
                "execution_time_seconds": exec_time,
                "pnl": pnl_value,
                "signal_count": signal_cnt,
            },
        )


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    order_id: Mapped[str] = mapped_column(
        String, ForeignKey("orders.id", ondelete="CASCADE")
    )
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    fee: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    fee_currency: Mapped[Optional[str]] = mapped_column(String(16))
    filled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    raw_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    order: Mapped["Order"] = relationship("Order", back_populates="fills")


@event.listens_for(Fill, "after_insert")
def _log_fill_created(mapper, connection, target: Fill) -> None:
    """
    Logs creation of a fill with key details.
    """
    logger.info(
        "fill_created",
        extra={
            "fill_id": target.id,
            "order_id": target.order_id,
            "quantity": float(target.quantity),
            "price": float(target.price),
            "fee": float(target.fee),
            "filled_at": target.filled_at.isoformat(),
        },
    )