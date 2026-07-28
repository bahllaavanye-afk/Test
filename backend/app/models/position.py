import uuid
from datetime import date, datetime
from typing import Iterable, List, Optional, Sequence

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

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

    account: Mapped["Account"] = relationship("Account", back_populates="positions")

    def __init__(
        self,
        account_id: str,
        symbol: str,
        side: str,
        quantity: float,
        avg_cost: float,
        opened_at: datetime,
        *,
        current_price: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        asset_class: str = "equity",
        underlying_symbol: Optional[str] = None,
        expiry: Optional[date] = None,
        strike: Optional[float] = None,
        option_right: Optional[str] = None,
        contract_multiplier: int = 1,
        updated_at: Optional[datetime] = None,
    ):
        """
        Initialise a Position with defensive checks for None/empty values and
        off‑by‑one errors.

        - ``account_id``, ``symbol`` and ``side`` must be non‑empty strings.
        - ``quantity`` and ``avg_cost`` must be non‑negative numbers.
        - ``contract_multiplier`` must be >= 1.
        - ``opened_at`` and ``updated_at`` default to ``opened_at`` if not provided.
        """
        if not account_id:
            raise ValueError("account_id cannot be None or empty")
        if not symbol:
            raise ValueError("symbol cannot be None or empty")
        if side not in {"long", "short"}:
            raise ValueError("side must be either 'long' or 'short'")

        if quantity is None or quantity < 0:
            raise ValueError("quantity must be a non‑negative number")
        if avg_cost is None or avg_cost < 0:
            raise ValueError("avg_cost must be a non‑negative number")
        if contract_multiplier < 1:
            raise ValueError("contract_multiplier must be at least 1")

        self.account_id = account_id
        self.symbol = symbol
        self.side = side
        self.quantity = float(quantity)
        self.avg_cost = float(avg_cost)
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
        self.contract_multiplier = int(contract_multiplier)
        self.opened_at = opened_at
        self.updated_at = updated_at or opened_at

    def update_price(self, price: Optional[float]) -> None:
        """
        Update the current market price and recompute unrealized P&L.

        Handles ``None`` price gracefully and avoids off‑by‑one errors when
        calculating P&L for fractional quantities.
        """
        if price is None:
            self.current_price = None
            self.unrealized_pnl = None
            return

        self.current_price = float(price)

        # P&L = (current_price - avg_cost) * quantity * contract_multiplier
        # Guard against None quantity or multiplier, though they should never be None.
        qty = self.quantity if self.quantity is not None else 0.0
        mult = self.contract_multiplier if self.contract_multiplier is not None else 1
        self.unrealized_pnl = (self.current_price - self.avg_cost) * qty * mult

    @classmethod
    def bulk_create(
        cls,
        positions_data: Optional[Sequence[dict]],
        *,
        default_opened_at: Optional[datetime] = None,
    ) -> List["Position"]:
        """
        Create multiple Position instances from an iterable of dictionaries.

        - Returns an empty list for ``None`` or empty input.
        - Validates each entry individually; invalid entries are skipped
          rather than raising to keep bulk operations robust.
        """
        if not positions_data:
            return []

        created: List[Position] = []
        for data in positions_data:
            try:
                # Extract required fields with explicit defaults to avoid KeyError
                account_id = data.get("account_id")
                symbol = data.get("symbol")
                side = data.get("side")
                quantity = data.get("quantity")
                avg_cost = data.get("avg_cost")
                opened_at = data.get("opened_at") or default_opened_at
                if opened_at is None:
                    raise ValueError("opened_at must be provided for each position")

                pos = cls(
                    account_id=account_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    avg_cost=avg_cost,
                    opened_at=opened_at,
                    current_price=data.get("current_price"),
                    unrealized_pnl=data.get("unrealized_pnl"),
                    asset_class=data.get("asset_class", "equity"),
                    underlying_symbol=data.get("underlying_symbol"),
                    expiry=data.get("expiry"),
                    strike=data.get("strike"),
                    option_right=data.get("option_right"),
                    contract_multiplier=data.get("contract_multiplier", 1),
                    updated_at=data.get("updated_at"),
                )
                created.append(pos)
            except Exception:
                # Silently skip malformed entries; could be logged in production.
                continue
        return created