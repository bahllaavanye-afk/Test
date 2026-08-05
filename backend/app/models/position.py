import uuid
from datetime import date, datetime
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Date, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("account_id", "symbol"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str] = mapped_column(String, ForeignKey("accounts.id", ondelete="CASCADE"))
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)   # long|short
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    avg_cost: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    current_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    unrealized_pnl: Mapped[float | None] = mapped_column(Numeric(18, 8))
    # Cross-desk tracking — one position shape for every desk (equity/crypto/option/...).
    asset_class: Mapped[str] = mapped_column(String(16), nullable=False, default="equity")
    underlying_symbol: Mapped[str | None] = mapped_column(String(32))  # options: the underlying
    expiry: Mapped[date | None] = mapped_column(Date)                  # options/futures
    strike: Mapped[float | None] = mapped_column(Numeric(18, 8))       # options
    option_right: Mapped[str | None] = mapped_column(String(4))        # call|put
    contract_multiplier: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    account: Mapped["Account"] = relationship("Account", back_populates="positions")


# ---------- Unit Tests ----------
# These tests focus on boundary conditions and default values for the Position model.
# They are lightweight and do not require a live database session.

def test_position_defaults():
    """Verify that default fields are correctly populated when not explicitly provided."""
    now = datetime.utcnow()
    pos = Position(
        account_id="acc_default",
        symbol="AAPL",
        side="long",
        quantity=10,
        avg_cost=150,
        opened_at=now,
        updated_at=now,
    )
    assert pos.asset_class == "equity", "Default asset_class should be 'equity'"
    assert pos.contract_multiplier == 1, "Default contract_multiplier should be 1"
    assert isinstance(pos.id, str) and len(pos.id) > 0, "ID should be a non‑empty string"


def test_position_boundary_quantity():
    """Edge case: zero quantity and zero avg_cost should be accepted without error."""
    now = datetime.utcnow()
    pos = Position(
        account_id="acc_zero",
        symbol="TSLA",
        side="short",
        quantity=0,
        avg_cost=0,
        opened_at=now,
        updated_at=now,
    )
    assert pos.quantity == 0, "Quantity of zero should be stored unchanged"
    assert pos.avg_cost == 0, "Avg cost of zero should be stored unchanged"


def test_position_expiry_none():
    """Edge case: expiry can be explicitly set to None for non‑expiring instruments."""
    now = datetime.utcnow()
    pos = Position(
        account_id="acc_no_expiry",
        symbol="SPY",
        side="long",
        quantity=5,
        avg_cost=300,
        opened_at=now,
        updated_at=now,
        expiry=None,
    )
    assert pos.expiry is None, "Expiry should remain None when not provided"
    assert pos.strike is None, "Strike should default to None when not set"
    assert pos.option_right is None, "Option right should default to None when not set"