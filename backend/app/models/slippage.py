import uuid
import logging
import math
from datetime import datetime
from sqlalchemy import String, ForeignKey, Numeric, DateTime, Float, event
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

logger = logging.getLogger(__name__)

def _validate_str(name: str, value: str | None, allow_empty: bool = False) -> None:
    if value is None:
        raise ValueError(f"'{name}' cannot be None")
    if not isinstance(value, str):
        raise ValueError(f"'{name}' must be a string, got {type(value).__name__}")
    if not allow_empty and value.strip() == "":
        raise ValueError(f"'{name}' cannot be an empty string")

def _validate_numeric(name: str, value: float | None, allow_none: bool = True, non_negative: bool = False) -> None:
    if value is None:
        if not allow_none:
            raise ValueError(f"'{name}' cannot be None")
        return
    if not isinstance(value, (int, float)):
        raise ValueError(f"'{name}' must be a numeric type, got {type(value).__name__}")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"'{name}' cannot be NaN")
    if non_negative and value < 0:
        raise ValueError(f"'{name}' cannot be negative, got {value}")

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
        """
        Validate inputs before initializing the SQLAlchemy model.
        """
        _validate_str("order_id", order_id)
        _validate_numeric("signal_price", signal_price)
        _validate_numeric("expected_price", expected_price)
        _validate_numeric("fill_price", fill_price)
        _validate_numeric("slippage_bps", slippage_bps)
        if execution_algo is not None:
            _validate_str("execution_algo", execution_algo, allow_empty=False)
        _validate_numeric("arrival_price", arrival_price)
        _validate_numeric("is_cost_bps", is_cost_bps)
        _validate_numeric("vwap_shortfall_bps", vwap_shortfall_bps)
        _validate_numeric("period_vwap", period_vwap)
        _validate_numeric("execution_duration_seconds", execution_duration_seconds, non_negative=True)

        super().__init__(
            order_id=order_id,
            signal_price=signal_price,
            expected_price=expected_price,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            execution_algo=execution_algo,
            arrival_price=arrival_price,
            is_cost_bps=is_cost_bps,
            vwap_shortfall_bps=vwap_shortfall_bps,
            period_vwap=period_vwap,
            execution_duration_seconds=execution_duration_seconds,
            created_at=created_at or datetime.utcnow(),
        )

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