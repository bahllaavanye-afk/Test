import uuid
from datetime import datetime, date
from typing import Optional, Dict, Any

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
    filled_qty: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    avg_fill_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    time_in_force: Mapped[str] = mapped_column(String(8), default="GTC")
    execution_algo: Mapped[Optional[str]] = mapped_column(String(32))
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
    trailing_stop_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    notional: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    bracket_parent_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("orders.id", ondelete="SET NULL")
    )
    risk_reward_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))

    # Cross-desk tracking — one order shape for every desk (equity/crypto/option/...).
    asset_class: Mapped[str] = mapped_column(
        String(16), nullable=False, default="equity"
    )
    underlying_symbol: Mapped[Optional[str]] = mapped_column(String(32))
    expiry: Mapped[Optional[date]] = mapped_column(Date)  # options/futures
    strike: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
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

    # -------------------------------------------------------------------------
    # Strategy signal quality helpers
    # -------------------------------------------------------------------------
    def is_entry_valid(self) -> bool:
        """Basic validation of order entry fields.

        Returns:
            bool: True if the order meets minimal entry criteria.
        """
        if self.quantity is None or self.quantity <= 0:
            return False
        if self.side not in {"buy", "sell"}:
            return False
        if self.order_type == "limit" and self.limit_price is None:
            return False
        if self.order_type == "stop" and self.stop_price is None:
            return False
        return True

    def confirm_entry(
        self, market_price: float, volatility: float = 0.0, tolerance: float = 0.01
    ) -> bool:
        """Secondary confirmation filter for entry signals.

        Args:
            market_price: Current market price of the symbol.
            volatility: Recent price volatility (e.g., standard deviation). Used to
                avoid entries during highly volatile periods.
            tolerance: Acceptable deviation for limit orders (default 1%).

        Returns:
            bool: True if the signal passes confirmation criteria.
        """
        # Simple volatility guard – can be tuned per strategy
        if volatility > 0.05:  # 5% volatility threshold as a placeholder
            return False

        if self.order_type == "limit":
            if self.limit_price is None:
                return False
            deviation = abs(market_price - float(self.limit_price)) / market_price
            return deviation <= tolerance
        if self.order_type == "stop":
            if self.stop_price is None:
                return False
            if self.side == "buy":
                return market_price >= float(self.stop_price)
            else:
                return market_price <= float(self.stop_price)
        # Market orders have no price confirmation needed
        return True

    def should_exit(self, market_price: float) -> bool:
        """Determine if the order should be exited based on TP/SL/Trailing logic.

        Args:
            market_price: Current market price of the symbol.

        Returns:
            bool: True if exit conditions are met.
        """
        # Take profit condition
        if self.take_profit_price is not None:
            if self.side == "buy" and market_price >= float(self.take_profit_price):
                return True
            if self.side == "sell" and market_price <= float(self.take_profit_price):
                return True

        # Stop loss condition
        if self.stop_loss_price is not None:
            if self.side == "buy" and market_price <= float(self.stop_loss_price):
                return True
            if self.side == "sell" and market_price >= float(self.stop_loss_price):
                return True

        # Trailing stop placeholder – actual implementation would track peak/ trough prices
        if self.trailing_stop_pct is not None:
            # In a production system we would maintain the highest (for long) or lowest
            # (for short) price seen since entry. Here we provide a deterministic stub.
            # Example logic:
            #   trail_price = peak_price * (1 - trailing_stop_pct/100) for long
            #   trail_price = trough_price * (1 + trailing_stop_pct/100) for short
            # Since we lack state, we return False to avoid premature exits.
            return False

        return False

    def compute_risk_reward_ratio(self) -> Optional[float]:
        """Calculate the risk‑reward ratio if TP and SL are defined.

        Returns:
            Optional[float]: Calculated ratio or None if insufficient data.
        """
        if (
            self.take_profit_price is None
            or self.stop_loss_price is None
            or self.avg_fill_price is None
        ):
            return None

        entry_price = float(self.avg_fill_price)
        tp = float(self.take_profit_price)
        sl = float(self.stop_loss_price)

        # Avoid division by zero
        if entry_price == sl:
            return None

        reward = abs(tp - entry_price)
        risk = abs(entry_price - sl)
        if risk == 0:
            return None

        return round(reward / risk, 4)


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