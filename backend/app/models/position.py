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
