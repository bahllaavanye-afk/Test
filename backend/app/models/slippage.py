import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class SlippageRecord(Base):
    __tablename__ = "slippage_records"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(String, ForeignKey("orders.id"), nullable=False, index=True)
    signal_price: Mapped[float | None] = mapped_column(Numeric(18, 8))   # price when signal fired
    expected_price: Mapped[float | None] = mapped_column(Numeric(18, 8)) # price when order submitted (arrival price)
    fill_price: Mapped[float | None] = mapped_column(Numeric(18, 8))     # actual fill price
    slippage_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))    # (fill-expected)/expected*10000
    execution_algo: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Item 5: Implementation Shortfall fields
    arrival_price: Mapped[float | None] = mapped_column(Numeric(18, 8))   # mid-price when order submitted
    is_cost_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))      # IS = (fill - arrival) / arrival * 10000
    vwap_shortfall_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))  # (fill - period_vwap) / period_vwap * 10000
    period_vwap: Mapped[float | None] = mapped_column(Numeric(18, 8))     # VWAP over execution period
    execution_duration_seconds: Mapped[float | None] = mapped_column(Float)  # time from submit to fill

    order: Mapped["Order"] = relationship("Order", back_populates="slippage")
