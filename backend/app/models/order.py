import uuid
from datetime import datetime
from datetime import date
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Date, Integer, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin


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
    broker_order_id: Mapped[str | None] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)   # buy|sell
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)  # market|limit|stop
    quantity: Mapped[float | None] = mapped_column(Numeric(18, 8))
    limit_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    stop_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    filled_qty: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    avg_fill_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    time_in_force: Mapped[str] = mapped_column(String(8), default="GTC")
    execution_algo: Mapped[str | None] = mapped_column(String(32))  # market|limit_first|twap|vwap
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    # Bracket / advanced order fields
    take_profit_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    stop_loss_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    trailing_stop_pct: Mapped[float | None] = mapped_column(Numeric(8, 4))  # e.g. 2.0 = 2%
    notional: Mapped[float | None] = mapped_column(Numeric(18, 8))  # buy $500 worth
    bracket_parent_id: Mapped[str | None] = mapped_column(String, ForeignKey("orders.id", ondelete="SET NULL"))
    risk_reward_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))

    # Cross-desk tracking — one order shape for every desk (equity/crypto/option/...).
    asset_class: Mapped[str] = mapped_column(String(16), nullable=False, default="equity")
    underlying_symbol: Mapped[str | None] = mapped_column(String(32))  # options: the underlying
    expiry: Mapped[date | None] = mapped_column(Date)                  # options/futures
    strike: Mapped[float | None] = mapped_column(Numeric(18, 8))       # options
    option_right: Mapped[str | None] = mapped_column(String(4))        # call|put
    contract_multiplier: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    account: Mapped["Account"] = relationship("Account", back_populates="orders")
    fills: Mapped[list["Fill"]] = relationship("Fill", back_populates="order")
    slippage: Mapped[list["SlippageRecord"]] = relationship("SlippageRecord", back_populates="order")


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(String, ForeignKey("orders.id", ondelete="CASCADE"))
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    fee: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    fee_currency: Mapped[str | None] = mapped_column(String(16))
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    order: Mapped["Order"] = relationship("Order", back_populates="fills")
