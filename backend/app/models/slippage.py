import uuid
import logging
from datetime import datetime
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Float, event
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

logger = logging.getLogger(__name__)

class SlippageRecord(Base):
    __tablename__ = "slippage_records"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id: Mapped[str] = mapped_column(String, ForeignKey("orders.id"), nullable=False, index=True)
    signal_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    expected_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    fill_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    slippage_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))
    execution_algo: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    arrival_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    is_cost_bps: MMapped[float | None] = mapped_column(Numeric(8, 4))
    vwap_shortfall_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))
    period_vwap: Mapped[float | None] = mapped_column(Numeric(18, 8))
    execution_duration_seconds: Mapped[float | None] = mapped_column(Float)

    order: Mapped["Order"] = relationship("Order", back_populates="slippage")

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

event.listen(SlippageRecord, "after_insert", _log_slippage_record)