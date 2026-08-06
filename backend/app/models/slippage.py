import uuid
import logging
import math
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
    is_cost_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))
    vwap_shortfall_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))
    period_vwap: Mapped[float | None] = mapped_column(Numeric(18, 8))
    execution_duration_seconds: Mapped[float | None] = mapped_column(Float)

    order: Mapped["Order"] = relationship("Order", back_populates="slippage")

    def __init__(
        self,
        order_id: str,
        signal_price: float | None = None,
        expected_price: float | None = None,
        fill_price: float | None = None,
        slippage_bps: float | None = None,
        execution_algo: str | None = None,
        arrival_price: float | None = None,
        is_cost_bps: float | None = None,
        vwap_shortfall_bps: float | None = None,
        period_vwap: float | None = None,
        execution_duration_seconds: float | None = None,
        created_at: datetime | None = None,
    ):
        # order_id validation
        if not isinstance(order_id, str) or not order_id.strip():
            raise ValueError("order_id must be a non‑empty string")
        self.order_id = order_id

        # numeric validation helper
        def _validate_numeric(name: str, value: float | None, allow_negative: bool = True):
            if value is None:
                return None
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a numeric type")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if not allow_negative and value < 0:
                raise ValueError(f"{name} cannot be negative")
            return float(value)

        self.signal_price = _validate_numeric("signal_price", signal_price)
        self.expected_price = _validate_numeric("expected_price", expected_price)
        self.fill_price = _validate_numeric("fill_price", fill_price)
        self.slippage_bps = _validate_numeric("slippage_bps", slippage_bps)
        self.execution_algo = execution_algo  # string, can be None or any value

        self.arrival_price = _validate_numeric("arrival_price", arrival_price)
        self.is_cost_bps = _validate_numeric("is_cost_bps", is_cost_bps)
        self.vwap_shortfall_bps = _validate_numeric("vwap_shortfall_bps", vwap_shortfall_bps)
        self.period_vwap = _validate_numeric("period_vwap", period_vwap)
        self.execution_duration_seconds = _validate_numeric(
            "execution_duration_seconds", execution_duration_seconds, allow_negative=False
        )

        self.created_at = created_at if created_at is not None else datetime.utcnow()


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