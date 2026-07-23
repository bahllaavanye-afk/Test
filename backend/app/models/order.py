import uuid
from datetime import datetime, date
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Date, Integer, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
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

    # ---------------------------
    # Validation helpers
    # ---------------------------
    _VALID_SIDES = {"buy", "sell"}
    _VALID_ORDER_TYPES = {"market", "limit", "stop"}
    _VALID_STATUSES = {"pending", "filled", "cancelled", "rejected", "partial"}
    _VALID_TIF = {"GTC", "IOC", "FOK"}
    _VALID_EXEC_ALGOS = {"market", "limit_first", "twap", "vwap", None}
    _VALID_ASSET_CLASSES = {"equity", "crypto", "option", "future", "forex"}
    _VALID_OPTION_RIGHTS = {"call", "put", None}

    @validates("side")
    def _validate_side(self, key: str, value: str) -> str:
        if value not in self._VALID_SIDES:
            raise ValueError(f"Invalid side '{value}'. Expected one of {self._VALID_SIDES}.")
        return value

    @validates("order_type")
    def _validate_order_type(self, key: str, value: str) -> str:
        if value not in self._VALID_ORDER_TYPES:
            raise ValueError(f"Invalid order_type '{value}'. Expected one of {self._VALID_ORDER_TYPES}.")
        return value

    @validates("status")
    def _validate_status(self, key: str, value: str) -> str:
        if value not in self._VALID_STATUSES:
            raise ValueError(f"Invalid status '{value}'. Expected one of {self._VALID_STATUSES}.")
        return value

    @validates("time_in_force")
    def _validate_time_in_force(self, key: str, value: str) -> str:
        if value not in self._VALID_TIF:
            raise ValueError(f"Invalid time_in_force '{value}'. Expected one of {self._VALID_TIF}.")
        return value

    @validates("execution_algo")
    def _validate_execution_algo(self, key: str, value: str | None) -> str | None:
        if value not in self._VALID_EXEC_ALGOS:
            raise ValueError(f"Invalid execution_algo '{value}'. Expected one of {self._VALID_EXEC_ALGOS}.")
        return value

    @validates("asset_class")
    def _validate_asset_class(self, key: str, value: str) -> str:
        if value not in self._VALID_ASSET_CLASSES:
            raise ValueError(f"Invalid asset_class '{value}'. Expected one of {self._VALID_ASSET_CLASSES}.")
        return value

    @validates("option_right")
    def _validate_option_right(self, key: str, value: str | None) -> str | None:
        if value not in self._VALID_OPTION_RIGHTS:
            raise ValueError(f"Invalid option_right '{value}'. Expected one of {self._VALID_OPTION_RIGHTS}.")
        return value

    @validates("contract_multiplier")
    def _validate_contract_multiplier(self, key: str, value: int) -> int:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("contract_multiplier must be a positive integer.")
        return value

    @validates("quantity", "limit_price", "stop_price", "take_profit_price", "stop_loss_price",
               "trailing_stop_pct", "notional", "risk_reward_ratio", "strike")
    def _validate_non_negative(self, key: str, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError(f"{key} must be non‑negative.")
        return value


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

    @validates("quantity")
    def _validate_quantity(self, key: str, value: float) -> float:
        if value <= 0:
            raise ValueError("Fill quantity must be greater than zero.")
        return value

    @validates("price")
    def _validate_price(self, key: str, value: float) -> float:
        if value <= 0:
            raise ValueError("Fill price must be greater than zero.")
        return value

    @validates("fee")
    def _validate_fee(self, key: str, value: float) -> float:
        if value < 0:
            raise ValueError("Fill fee cannot be negative.")
        return value