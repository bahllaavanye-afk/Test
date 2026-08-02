import uuid
import logging
from datetime import datetime

from sqlalchemy import String, Boolean, ForeignKey, Numeric, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.database import Base
from app.models.base import TimestampMixin

logger = logging.getLogger(__name__)


class Account(Base, TimestampMixin):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    broker: Mapped[str] = mapped_column(String(50), nullable=False)  # alpaca|tradestation|binance|polymarket
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    mode: Mapped[str] = mapped_column(String(10), nullable=False, default="paper")  # paper|live
    encrypted_key: Mapped[str | None] = mapped_column(String(1024))
    encrypted_secret: Mapped[str | None] = mapped_column(String(1024))
    extra_config: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped["User"] = relationship("User", back_populates="accounts")
    snapshots: Mapped[list["AccountSnapshot"]] = relationship(
        "AccountSnapshot", back_populates="account"
    )
    orders: Mapped[list["Order"]] = relationship("Order", back_populates="account")
    positions: Mapped[list["Position"]] = relationship(
        "Position", back_populates="account"
    )
    strategies: Mapped[list["Strategy"]] = relationship(
        "Strategy", back_populates="account"
    )

    @validates("broker")
    def validate_broker(self, key: str, value: str) -> str:
        allowed = {"alpaca", "tradestation", "binance", "polymarket"}
        if value not in allowed:
            logger.error(
                "Invalid broker value",
                extra={"field": key, "provided_value": value, "allowed": list(allowed)},
            )
            raise ValueError(f"Broker must be one of {allowed}, got '{value}'")
        return value

    @validates("mode")
    def validate_mode(self, key: str, value: str) -> str:
        allowed = {"paper", "live"}
        if value not in allowed:
            logger.error(
                "Invalid mode value",
                extra={"field": key, "provided_value": value, "allowed": list(allowed)},
            )
            raise ValueError(f"Mode must be one of {allowed}, got '{value}'")
        return value

    @validates("extra_config")
    def validate_extra_config(self, key: str, value: dict) -> dict:
        if not isinstance(value, dict):
            logger.error(
                "Invalid extra_config type",
                extra={"field": key, "provided_type": type(value).__name__},
            )
            raise TypeError("extra_config must be a dictionary")
        return value


class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    total_equity: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    cash: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    account: Mapped["Account"] = relationship("Account", back_populates="snapshots")

    @validates("total_equity", "cash", "unrealized_pnl")
    def validate_numeric_fields(self, key: str, value) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            logger.error(
                "Non-numeric value for financial field",
                extra={"field": key, "provided_value": value},
            )
            raise TypeError(f"{key} must be a numeric type") from exc

        if numeric_value < 0:
            logger.error(
                "Negative financial value detected",
                extra={"field": key, "value": numeric_value},
            )
            raise ValueError(f"{key} cannot be negative")
        return numeric_value

    @validates("raw_payload")
    def validate_raw_payload(self, key: str, value: dict) -> dict:
        if not isinstance(value, dict):
            logger.error(
                "Invalid raw_payload type",
                extra={"field": key, "provided_type": type(value).__name__},
            )
            raise TypeError("raw_payload must be a dictionary")
        return value