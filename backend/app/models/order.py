import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    String,
    ForeignKey,
    Numeric,
    DateTime,
    Date,
    Integer,
    JSON,
    Index,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)
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
    symbol: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True
    )
    side: Mapped[str] = mapped_column(
        String(8), nullable=False
    )  # buy|sell
    order_type: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # market|limit|stop
    quantity: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    limit_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    stop_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending"
    )
    filled_qty: Mapped[float] = mapped_column(
        Numeric(18, 8), default=0
    )
    avg_fill_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    time_in_force: Mapped[str] = mapped_column(
        String(8), default="GTC"
    )
    execution_algo: Mapped[Optional[str]] = mapped_column(
        String(32)
    )  # market|limit_first|twap|vwap
    submitted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    filled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    raw_payload: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict
    )

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
    fills: Mapped[List["Fill"]] = relationship(
        "Fill", back_populates="order", cascade="all, delete-orphan"
    )
    slippage: Mapped[List["SlippageRecord"]] = relationship(
        "SlippageRecord", back_populates="order", cascade="all, delete-orphan"
    )

    @validates("side")
    def validate_side(self, key: str, value: Optional[str]) -> str:
        """Ensure side is a non‑empty string and one of the allowed values."""
        if not value:
            raise ValueError("Order side cannot be None or empty.")
        normalized = value.strip().lower()
        if normalized not in {"buy", "sell"}:
            raise ValueError(f"Invalid order side '{value}'. Expected 'buy' or 'sell'.")
        return normalized

    @validates("order_type")
    def validate_order_type(self, key: str, value: Optional[str]) -> str:
        """Validate order_type against known types."""
        if not value:
            raise ValueError("order_type cannot be None or empty.")
        normalized = value.strip().lower()
        if normalized not in {"market", "limit", "stop"}:
            raise ValueError(
                f"Invalid order_type '{value}'. Expected 'market', 'limit' or 'stop'."
            )
        return normalized

    @validates("quantity")
    def validate_quantity(self, key: str, value: Optional[float]) -> float:
        """Treat None as zero and guard against negative quantities."""
        qty = 0.0 if value is None else float(value)
        if qty < 0:
            raise ValueError("Quantity cannot be negative.")
        return qty

    @validates("filled_qty")
    def validate_filled_qty(self, key: str, value: Optional[float]) -> float:
        """Ensure filled_qty is non‑negative and does not exceed quantity."""
        filled = 0.0 if value is None else float(value)
        if filled < 0:
            raise ValueError("filled_qty cannot be negative.")
        # quantity may be None (treated as 0); enforce off‑by‑one safety
        max_allowed = self.quantity if self.quantity is not None else 0.0
        if filled > max_allowed:
            # Clamp to max_allowed to avoid over‑fill inconsistencies
            return max_allowed
        return filled

    @property
    def remaining_quantity(self) -> float:
        """Quantity still to be filled; never negative."""
        total_qty = self.quantity if self.quantity is not None else 0.0
        remaining = total_qty - self.filled_qty
        return max(0.0, float(remaining))


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    order_id: Mapped[str] = mapped_column(
        String, ForeignKey("orders.id", ondelete="CASCADE")
    )
    quantity: Mapped[float] = mapped_column(
        Numeric(18, 8), nullable=False
    )
    price: Mapped[float] = mapped_column(
        Numeric(18, 8), nullable=False
    )
    fee: Mapped[float] = mapped_column(
        Numeric(18, 8), default=0
    )
    fee_currency: Mapped[Optional[str]] = mapped_column(
        String(16)
    )
    filled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    raw_payload: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict
    )

    order: Mapped["Order"] = relationship(
        "Order", back_populates="fills"
    )

    @validates("quantity", "price", "fee")
    def validate_non_negative(self, key: str, value: Optional[float]) -> float:
        """Guard against None and negative numeric values for fill attributes."""
        val = 0.0 if value is None else float(value)
        if val < 0:
            raise ValueError(f"{key} cannot be negative.")
        return val