import uuid
from datetime import datetime, date
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Date, Integer, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models.base import TimestampMixin

"""SQLAlchemy ORM models for order management.

This module defines the core data structures used to persist order and fill
information within the QuantEdge trading platform. The models are deliberately
lightweight and rely on SQLAlchemy's declarative mapping; all business logic
resides elsewhere in the service layer.
"""


class Order(Base, TimestampMixin):
    """Represents a trading order placed through a broker.

    The table stores both simple market/limit orders and more complex bracket
    orders. Each row is linked to an ``Account`` and optionally to a ``Strategy``.
    Fields are typed using SQLAlchemy ``Mapped`` annotations to provide static
    type checking and IDE assistance.

    Attributes
    ----------
    id: str
        Primary key; a UUID string generated automatically.
    account_id: str
        Foreign key to the owning account.
    strategy_id: Optional[str]
        Foreign key to the strategy that generated the order; may be null.
    broker_order_id: Optional[str]
        Identifier returned by the broker; indexed for fast lookup.
    symbol: str
        Trading symbol (e.g., ``AAPL`` or ``BTCUSD``).
    side: str
        Order side – ``"buy"`` or ``"sell"``.
    order_type: str
        Type of order – ``"market"``, ``"limit"``, ``"stop"``, etc.
    quantity: Optional[float]
        Number of units to trade; stored with high precision.
    limit_price: Optional[float]
        Limit price for limit orders.
    stop_price: Optional[float]
        Stop price for stop orders.
    status: str
        Current order status; defaults to ``"pending"``.
    filled_qty: float
        Quantity that has been filled; defaults to ``0``.
    avg_fill_price: Optional[float]
        Volume‑weighted average price of filled quantity.
    time_in_force: str
        Order time‑in‑force policy; defaults to ``"GTC"`` (good‑til‑canceled).
    execution_algo: Optional[str]
        Execution algorithm identifier (e.g., ``"twap"``, ``"vwap"``).
    submitted_at: Optional[datetime]
        Timestamp when the order was submitted to the broker.
    filled_at: Optional[datetime]
        Timestamp when the order was fully filled.
    cancelled_at: Optional[datetime]
        Timestamp when the order was cancelled.
    raw_payload: dict
        Raw broker response payload; useful for debugging and audit trails.
    take_profit_price: Optional[float]
        Price level for take‑profit (bracket order).
    stop_loss_price: Optional[float]
        Price level for stop‑loss (bracket order).
    trailing_stop_pct: Optional[float]
        Trailing stop percentage (e.g., ``2.0`` for 2 %).
    notional: Optional[float]
        Notional value for orders expressed in currency rather than quantity.
    bracket_parent_id: Optional[str]
        Reference to the parent order for bracket structures.
    risk_reward_ratio: Optional[float]
        Desired risk‑to‑reward ratio for the order.
    asset_class: str
        Asset class classification; defaults to ``"equity"``.
    underlying_symbol: Optional[str]
        Underlying symbol for derivatives (options, futures).
    expiry: Optional[date]
        Expiration date for options or futures contracts.
    strike: Optional[float]
        Strike price for options.
    option_right: Optional[str]
        Option right type – ``"call"`` or ``"put"``.
    contract_multiplier: int
        Contract multiplier; defaults to ``1``.
    account: Account
        ORM relationship to the owning ``Account``.
    fills: List[Fill]
        Collection of ``Fill`` records associated with the order.
    slippage: List[SlippageRecord]
        Collection of slippage records for the order.
    """

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
    """Represents a partial execution (fill) of an ``Order``.

    Each fill records the quantity, execution price, applicable fees, and the
    timestamp when the fill occurred. The raw broker payload is retained for
    auditability.

    Attributes
    ----------
    id: str
        Primary key; a UUID string generated automatically.
    order_id: str
        Foreign key linking back to the parent ``Order``.
    quantity: float
        Quantity filled in this execution.
    price: float
        Execution price for the filled quantity.
    fee: float
        Fee charged by the broker for this fill.
    fee_currency: Optional[str]
        Currency of the fee (e.g., ``"USD"``).
    filled_at: datetime
        Timestamp when the fill was recorded.
    raw_payload: dict
        Raw broker response payload for the fill.
    order: Order
        ORM relationship back to the parent ``Order``.
    """

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