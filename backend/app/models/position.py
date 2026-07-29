import uuid
from datetime import date, datetime
from typing import Optional, Any

from sqlalchemy import (
    String,
    ForeignKey,
    Numeric,
    DateTime,
    Date,
    Integer,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)
from app.database import Base


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("account_id", "symbol"),)

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.id", ondelete="CASCADE")
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)  # long|short
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    avg_cost: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    current_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    unrealized_pnl: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    # Cross-desk tracking — one position shape for every desk (equity/crypto/option/...).
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
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    account: Mapped["Account"] = relationship(
        "Account", back_populates="positions"
    )

    def __init__(
        self,
        account_id: str,
        symbol: str,
        side: str,
        quantity: Optional[float] = None,
        avg_cost: Optional[float] = None,
        current_price: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        asset_class: str = "equity",
        underlying_symbol: Optional[str] = None,
        expiry: Optional[date] = None,
        strike: Optional[float] = None,
        option_right: Optional[str] = None,
        contract_multiplier: Optional[int] = None,
        opened_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> None:
        """
        Initialise a Position instance with defensive handling for edge cases.

        - None inputs for required numeric fields default to 0.
        - Empty strings for required textual fields raise ValueError.
        - Off‑by‑one issues for contract_multiplier are guarded (minimum 1).
        """
        if not account_id:
            raise ValueError("account_id must be a non‑empty string")
        if not symbol:
            raise ValueError("symbol must be a non‑empty string")
        self.account_id = account_id
        self.symbol = symbol

        self.side = side  # validation performed by SQLAlchemy validator

        # Defensive defaults for numeric fields
        self.quantity = float(quantity) if quantity is not None else 0.0
        self.avg_cost = float(avg_cost) if avg_cost is not None else 0.0
        self.current_price = (
            float(current_price) if current_price is not None else None
        )
        self.unrealized_pnl = (
            float(unrealized_pnl) if unrealized_pnl is not None else None
        )

        self.asset_class = asset_class or "equity"
        self.underlying_symbol = underlying_symbol
        self.expiry = expiry
        self.strike = float(strike) if strike is not None else None
        self.option_right = option_right

        # Ensure contract_multiplier is at least 1 (off‑by‑one protection)
        self.contract_multiplier = (
            int(contract_multiplier) if contract_multiplier and contract_multiplier > 0 else 1
        )

        now = datetime.utcnow()
        self.opened_at = opened_at or now
        self.updated_at = updated_at or now

    @validates("side")
    def _validate_side(self, key: str, value: Any) -> str:
        if value not in {"long", "short"}:
            raise ValueError("side must be either 'long' or 'short'")
        return value

    @validates("quantity")
    def _validate_quantity(self, key: str, value: Any) -> float:
        try:
            qty = float(value)
        except (TypeError, ValueError):
            raise ValueError("quantity must be a numeric value")
        if qty < 0:
            raise ValueError("quantity cannot be negative")
        return qty

    @validates("avg_cost")
    def _validate_avg_cost(self, key: str, value: Any) -> float:
        try:
            cost = float(value)
        except (TypeError, ValueError):
            raise ValueError("avg_cost must be a numeric value")
        if cost < 0:
            raise ValueError("avg_cost cannot be negative")
        return cost

    @validates("contract_multiplier")
    def _validate_contract_multiplier(self, key: str, value: Any) -> int:
        try:
            mult = int(value)
        except (TypeError, ValueError):
            raise ValueError("contract_multiplier must be an integer")
        if mult < 1:
            # Guard against off‑by‑one errors that could set this to 0
            raise ValueError("contract_multiplier must be at least 1")
        return mult

    def __repr__(self) -> str:
        return (
            f"Position(id={self.id!r}, account_id={self.account_id!r}, "
            f"symbol={self.symbol!r}, side={self.side!r}, quantity={self.quantity!r}, "
            f"avg_cost={self.avg_cost!r})"
        )