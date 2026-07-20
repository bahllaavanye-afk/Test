import uuid
from datetime import datetime, date
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Date, Integer, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin

# Constants for field lengths and defaults
BROKER_ORDER_ID_MAX_LEN = 128
SYMBOL_MAX_LEN = 32
SIDE_MAX_LEN = 8
ORDER_TYPE_MAX_LEN = 16
STATUS_MAX_LEN = 16
TIME_IN_FORCE_MAX_LEN = 8
EXEC_ALGO_MAX_LEN = 32
ASSET_CLASS_MAX_LEN = 16
OPTION_RIGHT_MAX_LEN = 4

STATUS_DEFAULT = "pending"
TIME_IN_FORCE_DEFAULT = "GTC"
ASSET_CLASS_DEFAULT = "equity"
CONTRACT_MULTIPLIER_DEFAULT = 1

NUMERIC_PRECISION = (18, 8)
PCT_PRECISION = (8, 4)


class Order(Base, TimestampMixin):
    __tablename__ = "orders"
    __table_args__ = (
        # Composite indexes for the most common query patterns
        Index("ix_orders_account_status", "account_id", "status"),
        Index("ix_orders_account_created", "account_id", "created_at"),
        Index("ix_orders_symbol_status", "symbol", "status"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str] = mapped_column(String, ForeignKey("accounts.id", ondelete="CASCADE"))
    strategy_id: Mapped[str | None] = mapped_column(String, ForeignKey("strategies.id", ondelete="SET NULL"))
    broker_order_id: Mapped[str | None] = mapped_column(String(BROKER_ORDER_ID_MAX_LEN), index=True)
    symbol: Mapped[str] = mapped_column(String(SYMBOL_MAX_LEN), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(SIDE_MAX_LEN), nullable=False)   # buy|sell
    order_type: Mapped[str] = mapped_column(String(ORDER_TYPE_MAX_LEN), nullable=False)  # market|limit|stop
    quantity: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_PRECISION))
    limit_price: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_PRECISION))
    stop_price: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_PRECISION))
    status: Mapped[str] = mapped_column(String(STATUS_MAX_LEN), nullable=False, default=STATUS_DEFAULT)
    filled_qty: Mapped[float] = mapped_column(Numeric(*NUMERIC_PRECISION), default=0)
    avg_fill_price: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_PRECISION))
    time_in_force: Mapped[str] = mapped_column(String(TIME_IN_FORCE_MAX_LEN), default=TIME_IN_FORCE_DEFAULT)
    execution_algo: Mapped[str | None] = mapped_column(String(EXEC_ALGO_MAX_LEN))  # market|limit_first|twap|vwap
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    # Bracket / advanced order fields
    take_profit_price: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_PRECISION))
    stop_loss_price: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_PRECISION))
    trailing_stop_pct: Mapped[float | None] = mapped_column(Numeric(*PCT_PRECISION))  # e.g. 2.0 = 2%
    notional: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_PRECISION))  # buy $500 worth
    bracket_parent_id: Mapped[str | None] = mapped_column(String, ForeignKey("orders.id", ondelete="SET NULL"))
    risk_reward_ratio: Mapped[float | None] = mapped_column(Numeric(*PCT_PRECISION))

    # Cross-desk tracking — one order shape for every desk (equity/crypto/option/...).
    asset_class: Mapped[str] = mapped_column(String(ASSET_CLASS_MAX_LEN), nullable=False, default=ASSET_CLASS_DEFAULT)
    underlying_symbol: Mapped[str | None] = mapped_column(String(SYMBOL_MAX_LEN))  # options: the underlying
    expiry: Mapped[date | None] = mapped_column(Date)                  # options/futures
    strike: Mapped[float | None] = mapped_column(Numeric(*NUMERIC_PRECISION))       # options
    option_right: Mapped[str | None] = mapped_column(String(OPTION_RIGHT_MAX_LEN))        # call|put
    contract_multiplier: Mapped[int] = mapped_column(Integer, nullable=False, default=CONTRACT_MULTIPLIER_DEFAULT)

    account: Mapped["Account"] = relationship("Account", back_populates="orders")
    fills: Mapped[list["Fill"]] = relationship("Fill", back_populates="order")
    slippage: Mapped[list["SlippageRecord"]] = relationship("SlippageRecord", back_populates="order")


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(String, ForeignKey("orders.id", ondelete="CASCADE"))
    quantity: Mapped[float] = mapped_column(Numeric(*NUMERIC_PRECISION), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(*NUMERIC_PRECISION), nullable=False)
    fee: Mapped[float] = mapped_column(Numeric(*NUMERIC_PRECISION), default=0)
    fee_currency: Mapped[str | None] = mapped_column(String(16))
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    order: Mapped["Order"] = relationship("Order", back_populates="fills")