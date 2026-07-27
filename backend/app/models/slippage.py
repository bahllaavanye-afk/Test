import uuid
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import String, ForeignKey, Numeric, DateTime, Float, event
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

logger = logging.getLogger(__name__)


class SlippageRecord(Base):
    __tablename__ = "slippage_records"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    order_id: Mapped[str] = mapped_column(
        String, ForeignKey("orders.id"), nullable=False, index=True
    )
    signal_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    expected_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    fill_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    slippage_bps: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    execution_algo: Mapped[Optional[str]] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Implementation Shortfall fields
    arrival_price: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    is_cost_bps: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    vwap_shortfall_bps: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    period_vwap: Mapped[Optional[float]] = mapped_column(Numeric(18, 8))
    execution_duration_seconds: Mapped[Optional[float]] = mapped_column(Float)

    order: Mapped["Order"] = relationship("Order", back_populates="slippage")

    # --------------------------------------------------------------------- #
    # Helper / computed properties
    # --------------------------------------------------------------------- #
    @property
    def has_valid_prices(self) -> bool:
        """Return True if the essential price fields are present and sensible."""
        return (
            self.fill_price is not None
            and self.expected_price is not None
            and self.expected_price > 0
        )

    def compute_slippage_bps(self) -> Optional[float]:
        """Calculate slippage in basis points if possible."""
        if not self.has_valid_prices:
            return None
        return ((self.fill_price - self.expected_price) / self.expected_price) * 10_000

    def compute_is_cost_bps(self) -> Optional[float]:
        """Implementation shortfall based on arrival price."""
        if self.arrival_price is None or self.arrival_price == 0 or self.fill_price is None:
            return None
        return ((self.fill_price - self.arrival_price) / self.arrival_price) * 10_000

    def compute_vwap_shortfall_bps(self) -> Optional[float]:
        """Shortfall relative to period VWAP."""
        if self.period_vwap is None or self.period_vwap == 0 or self.fill_price is None:
            return None
        return ((self.fill_price - self.period_vwap) / self.period_vwap) * 10_000

    def is_excessive_slippage(self, threshold_bps: float = 50.0) -> bool:
        """
        Determine if slippage exceeds a configurable threshold.
        This aids downstream exit‑logic decisions.
        """
        if self.slippage_bps is None:
            return False
        return abs(self.slippage_bps) > threshold_bps


def _validate_and_compute_slippage(mapper, connection, target: SlippageRecord):
    """
    Validate essential fields and auto‑populate derived metrics.
    This tightens entry conditions for the record and ensures
    downstream analytics have consistent data.
    """
    # Validate that required price fields are present and non‑negative
    if target.fill_price is None or target.expected_price is None:
        raise ValueError(
            f"SlippageRecord {target.id} must have both fill_price and expected_price defined."
        )
    if target.expected_price <= 0:
        raise ValueError(
            f"expected_price must be positive for SlippageRecord {target.id}."
        )

    # Auto‑compute slippage if not supplied
    if target.slippage_bps is None:
        target.slippage_bps = target.compute_slippage_bps()

    # Auto‑compute implementation shortfall metrics
    if target.is_cost_bps is None:
        target.is_cost_bps = target.compute_is_cost_bps()
    if target.vwap_shortfall_bps is None:
        target.vwap_shortfall_bps = target.compute_vwap_shortfall_bps()


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


# Register event listeners to enforce validation and logging.
event.listen(SlippageRecord, "before_insert", _validate_and_compute_slippage)
event.listen(SlippageRecord, "before_update", _validate_and_compute_slippage)
event.listen(SlippageRecord, "after_insert", _log_slippage_record)