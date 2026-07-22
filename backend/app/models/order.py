import uuid
import logging
from datetime import datetime, date
from typing import Any, Dict

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

logger = logging.getLogger(__name__)


class OrderValidationError(ValueError):
    """Exception raised when an Order model receives invalid data."""


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
    strategy_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("strategies.id", ondelete="SET NULL")
    )
    broker_order_id: Mapped[str | None] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # buy|sell
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
    raw_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Bracket / advanced order fields
    take_profit_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    stop_loss_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    trailing_stop_pct: Mapped[float | None] = mapped_column(Numeric(8, 4))  # e.g. 2.0 = 2%
    notional: Mapped[float | None] = mapped_column(Numeric(18, 8))  # buy $500 worth
    bracket_parent_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("orders.id", ondelete="SET NULL")
    )
    risk_reward_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))

    # Cross-desk tracking — one order shape for every desk (equity/crypto/option/...).
    asset_class: Mapped[str] = mapped_column(
        String(16), nullable=False, default="equity"
    )
    underlying_symbol: Mapped[str | None] = mapped_column(String(32))  # options: the underlying
    expiry: Mapped[date | None] = mapped_column(Date)  # options/futures
    strike: Mapped[float | None] = mapped_column(Numeric(18, 8))  # options
    option_right: Mapped[str | None] = mapped_column(String(4))  # call|put
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
    # Validation helpers – raise specific errors and log details
    # --------------------------------------------------------------------- #
    @validates("side")
    def validate_side(self, key: str, value: str) -> str:
        if value not in {"buy", "sell"}:
            logger.error(
                "Invalid side for Order",
                order_id=self.id,
                field=key,
                invalid_value=value,
                expected="buy|sell",
            )
            raise OrderValidationError(
                f"Order side must be 'buy' or 'sell', got '{value}'."
            )
        return value

    @validates("order_type")
    def validate_order_type(self, key: str, value: str) -> str:
        if value not in {"market", "limit", "stop"}:
            logger.error(
                "Invalid order_type for Order",
                order_id=self.id,
                field=key,
                invalid_value=value,
                expected="market|limit|stop",
            )
            raise OrderValidationError(
                f"Order type must be one of 'market', 'limit', 'stop', got '{value}'."
            )
        return value

    @validates("time_in_force")
    def validate_time_in_force(self, key: str, value: str) -> str:
        if value not in {"GTC", "IOC", "FOK"}:
            logger.error(
                "Invalid time_in_force for Order",
                order_id=self.id,
                field=key,
                invalid_value=value,
                expected="GTC|IOC|FOK",
            )
            raise OrderValidationError(
                f"Time in force must be 'GTC', 'IOC', or 'FOK', got '{value}'."
            )
        return value

    @validates("asset_class")
    def validate_asset_class(self, key: str, value: str) -> str:
        if value not in {"equity", "crypto", "option", "future", "forex"}:
            logger.error(
                "Invalid asset_class for Order",
                order_id=self.id,
                field=key,
                invalid_value=value,
                expected="equity|crypto|option|future|forex",
            )
            raise OrderValidationError(
                f"Asset class must be one of 'equity', 'crypto', 'option', 'future', 'forex', got '{value}'."
            )
        return value

    @validates("option_right")
    def validate_option_right(self, key: str, value: str | None) -> str | None:
        if value is not None and value not in {"call", "put"}:
            logger.error(
                "Invalid option_right for Order",
                order_id=self.id,
                field=key,
                invalid_value=value,
                expected="call|put",
            )
            raise OrderValidationError(
                f"Option right must be 'call' or 'put' when set, got '{value}'."
            )
        return value

    @validates("quantity", "limit_price", "stop_price", "filled_qty", "avg_fill_price")
    def validate_numeric_fields(self, key: str, value: float | None) -> float | None:
        if value is not None and value < 0:
            logger.error(
                "Negative numeric value for Order",
                order_id=self.id,
                field=key,
                invalid_value=value,
            )
            raise OrderValidationError(
                f"Field '{key}' cannot be negative, got {value}."
            )
        return value


class FillValidationError(ValueError):
    """Exception raised when a Fill model receives invalid data."""


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
    fee_currency: Mapped[str | None] = mapped_column(String(16))
    filled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    raw_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    order: Mapped["Order"] = relationship("Order", back_populates="fills")

    @validates("quantity", "price", "fee")
    def validate_positive_numbers(self, key: str, value: float) -> float:
        if value < 0:
            logger.error(
                "Negative numeric value for Fill",
                fill_id=self.id,
                field=key,
                invalid_value=value,
            )
            raise FillValidationError(
                f"Field '{key}' must be non‑negative, got {value}."
            )
        return value

    @validates("fee_currency")
    def validate_fee_currency(self, key: str, value: str | None) -> str | None:
        if value is not None and not isinstance(value, str):
            logger.error(
                "Invalid fee_currency type for Fill",
                fill_id=self.id,
                field=key,
                invalid_value=value,
                expected_type="str",
            )
            raise FillValidationError(
                f"fee_currency must be a string when provided, got {type(value)}."
            )
        return value