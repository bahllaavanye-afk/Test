import uuid
import logging
from datetime import date, datetime
from sqlalchemy import (
    String,
    ForeignKey,
    Numeric,
    DateTime,
    Date,
    Integer,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from app.database import Base

logger = logging.getLogger(__name__)


def _generate_uuid() -> str:
    """Generate a UUID string with error handling."""
    try:
        return str(uuid.uuid4())
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to generate UUID for Position",
            extra={"exception_type": type(exc).__name__, "exception_msg": str(exc)},
        )
        raise


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("account_id", "symbol"),)

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=_generate_uuid
    )
    account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.id", ondelete="CASCADE")
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # long|short
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    avg_cost: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    current_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    unrealized_pnl: Mapped[float | None] = mapped_column(Numeric(18, 8))
    # Cross-desk tracking — one position shape for every desk (equity/crypto/option/...).
    asset_class: Mapped[str] = mapped_column(
        String(16), nullable=False, default="equity"
    )
    underlying_symbol: Mapped[str | None] = mapped_column(String(32))
    expiry: Mapped[date | None] = mapped_column(Date)  # options/futures
    strike: Mapped[float | None] = mapped_column(Numeric(18, 8))
    option_right: Mapped[str | None] = mapped_column(String(4))
    contract_multiplier: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    account: Mapped["Account"] = relationship(
        "Account", back_populates="positions"
    )

    @validates("side")
    def validate_side(self, key: str, value: str) -> str:
        if value not in {"long", "short"}:
            logger.error(
                "Invalid side value for Position",
                extra={"field": key, "invalid_value": value},
            )
            raise ValueError(f"side must be 'long' or 'short', got '{value}'")
        return value

    @validates("quantity")
    def validate_quantity(self, key: str, value: float) -> float:
        if value < 0:
            logger.error(
                "Negative quantity for Position",
                extra={"field": key, "invalid_value": value},
            )
            raise ValueError("quantity cannot be negative")
        return value

    @validates("asset_class")
    def validate_asset_class(self, key: str, value: str) -> str:
        allowed = {"equity", "crypto", "option", "future", "fx"}
        if value not in allowed:
            logger.error(
                "Invalid asset_class for Position",
                extra={"field": key, "invalid_value": value, "allowed": list(allowed)},
            )
            raise ValueError(
                f"asset_class must be one of {allowed}, got '{value}'"
            )
        return value

    @validates("option_right")
    def validate_option_right(self, key: str, value: str | None) -> str | None:
        if value is not None and value not in {"call", "put"}:
            logger.error(
                "Invalid option_right for Position",
                extra={"field": key, "invalid_value": value},
            )
            raise ValueError("option_right must be 'call', 'put', or None")
        return value

    @validates("contract_multiplier")
    def validate_contract_multiplier(self, key: str, value: int) -> int:
        if value <= 0:
            logger.error(
                "Non-positive contract_multiplier for Position",
                extra={"field": key, "invalid_value": value},
            )
            raise ValueError("contract_multiplier must be a positive integer")
        return value