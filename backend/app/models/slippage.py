import uuid
import logging
from datetime import datetime
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Float, event
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from app.database import Base

logger = logging.getLogger(__name__)

class SlippageRecord(Base):
    __tablename__ = "slippage_records"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(String, ForeignKey("orders.id"), nullable=False, index=True)
    signal_price: Mapped[float | None] = mapped_column(Numeric(18, 8))   # price when signal fired
    expected_price: Mapped[float | None] = mapped_column(Numeric(18, 8)) # price when order submitted (arrival price)
    fill_price: Mapped[float | None] = mapped_column(Numeric(18, 8))     # actual fill price
    slippage_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))    # (fill-expected)/expected*10000
    execution_algo: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # Item 5: Implementation Shortfall fields
    arrival_price: Mapped[float | None] = mapped_column(Numeric(18, 8))   # mid-price when order submitted
    is_cost_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))      # IS = (fill - arrival) / arrival * 10000
    vwap_shortfall_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))  # (fill - period_vwap) / period_vwap * 10000
    period_vwap: Mapped[float | None] = mapped_column(Numeric(18, 8))     # VWAP over execution period
    execution_duration_seconds: Mapped[float | None] = mapped_column(Float)  # time from submit to fill

    order: Mapped["Order"] = relationship("Order", back_populates="slippage")

    @validates("signal_price", "expected_price", "fill_price", "arrival_price", "period_vwap")
    def _validate_non_negative(self, key, value):
        """Ensure price-like fields are non‑negative when provided."""
        if value is not None and value < 0:
            raise ValueError(f"{key} must be non‑negative, got {value}")
        return value

    def _compute_derived_metrics(self):
        """Calculate slippage and short‑fall metrics if sufficient data is present."""
        if self.fill_price is not None and self.expected_price:
            self.slippage_bps = ((self.fill_price - self.expected_price) / self.expected_price) * 10000
        if self.fill_price is not None and self.arrival_price:
            self.is_cost_bps = ((self.fill_price - self.arrival_price) / self.arrival_price) * 10000
        if self.fill_price is not None and self.period_vwap:
            self.vwap_shortfall_bps = ((self.fill_price - self.period_vwap) / self.period_vwap) * 10000

def _prepare_slippage_record(mapper, connection, target: SlippageRecord):
    """Event hook to compute derived metrics before persisting."""
    target._compute_derived_metrics()

def _log_slippage_record(mapper, connection, target: SlippageRecord):
    """
    Structured logging for SlippageRecord creation.
    Logs key metrics at INFO level.
    """
    logger.info(
        "SlippageRecord created",
        extra={
            "record_id": target.id,
            "order_id": target.order_id,
            "signal_price": float(target.signal_price) if target.signal_price is not None else None,
            "expected_price": float(target.expected_price) if target.expected_price is not None else None,
            "fill_price": float(target.fill_price) if target.fill_price is not None else None,
            "slippage_bps": float(target.slippage_bps) if target.slippage_bps is not None else None,
            "execution_algo": target.execution_algo,
            "execution_duration_seconds": target.execution_duration_seconds,
            "is_cost_bps": float(target.is_cost_bps) if target.is_cost_bps is not None else None,
            "vwap_shortfall_bps": float(target.vwap_shortfall_bps) if target.vwap_shortfall_bps is not None else None,
        },
    )

event.listen(SlippageRecord, "before_insert", _prepare_slippage_record)
event.listen(SlippageRecord, "after_insert", _log_slippage_record)