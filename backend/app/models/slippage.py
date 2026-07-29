import uuid
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import String, ForeignKey, Numeric, DateTime, Float, event
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pydantic import BaseModel, Field, validator

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
    signal_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    expected_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    fill_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    slippage_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))
    execution_algo: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Implementation Shortfall fields
    arrival_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    is_cost_bps: Mapped[float | None] = mapped_column(Numeric(8, 4))
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
            "signal_price": float(target.signal_price)
            if target.signal_price is not None
            else None,
            "expected_price": float(target.expected_price)
            if target.expected_price is not None
            else None,
            "fill_price": float(target.fill_price)
            if target.fill_price is not None
            else None,
            "slippage_bps": float(target.slippage_bps)
            if target.slippage_bps is not None
            else None,
            "execution_algo": target.execution_algo,
            "execution_duration_seconds": target.execution_duration_seconds,
            "is_cost_bps": float(target.is_cost_bps)
            if target.is_cost_bps is not None
            else None,
            "vwap_shortfall_bps": float(target.vwap_shortfall_bps)
            if target.vwap_shortfall_bps is not None
            else None,
        },
    )


event.listen(SlippageRecord, "after_insert", _log_slippage_record)


class SlippageRecordSchema(BaseModel):
    """
    Pydantic schema for SlippageRecord.
    Provides field descriptions, examples, and basic validation.
    """

    id: str = Field(
        ...,
        description="Unique identifier for the slippage record.",
        example="d290f1ee-6c54-4b01-90e6-d701748f0851",
    )
    order_id: str = Field(
        ...,
        description="Reference to the associated order.",
        example="a1b2c3d4-5678-90ab-cdef-1234567890ab",
    )
    signal_price: Optional[float] = Field(
        None,
        description="Price at which the original trading signal was generated.",
        example=102.34,
    )
    expected_price: Optional[float] = Field(
        None,
        description="Price expected when the order was submitted (arrival price).",
        example=102.50,
    )
    fill_price: Optional[float] = Field(
        None,
        description="Actual execution price of the order.",
        example=102.55,
    )
    slippage_bps: Optional[float] = Field(
        None,
        description="Slippage in basis points: (fill - expected) / expected * 10,000.",
        example=4.88,
    )
    execution_algo: Optional[str] = Field(
        None,
        description="Identifier of the execution algorithm used.",
        example="TWAP",
        max_length=32,
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the slippage record was created.",
        example="2024-01-15T12:34:56Z",
    )
    arrival_price: Optional[float] = Field(
        None,
        description="Mid-price when the order was submitted.",
        example=102.48,
    )
    is_cost_bps: Optional[float] = Field(
        None,
        description="Implementation shortfall in basis points: (fill - arrival) / arrival * 10,000.",
        example=6.86,
    )
    vwap_shortfall_bps: Optional[float] = Field(
        None,
        description="VWAP shortfall in basis points: (fill - period_vwap) / period_vwap * 10,000.",
        example=2.15,
    )
    period_vwap: Optional[float] = Field(
        None,
        description="VWAP over the execution period.",
        example=102.52,
    )
    execution_duration_seconds: Optional[float] = Field(
        None,
        description="Time elapsed from order submission to fill, in seconds.",
        example=0.75,
    )

    @validator("slippage_bps", always=True)
    def validate_slippage_bps(cls, v, values):
        fill = values.get("fill_price")
        expected = values.get("expected_price")
        if fill is not None and expected not in (None, 0):
            computed = (fill - expected) / expected * 10000
            if v is not None and abs(v - computed) > 1e-2:
                raise ValueError(
                    "slippage_bps does not match fill_price and expected_price"
                )
            return round(computed, 4)
        return v

    @validator("is_cost_bps", always=True)
    def validate_is_cost_bps(cls, v, values):
        fill = values.get("fill_price")
        arrival = values.get("arrival_price")
        if fill is not None and arrival not in (None, 0):
            computed = (fill - arrival) / arrival * 10000
            if v is not None and abs(v - computed) > 1e-2:
                raise ValueError(
                    "is_cost_bps does not match fill_price and arrival_price"
                )
            return round(computed, 4)
        return v

    @validator("vwap_shortfall_bps", always=True)
    def validate_vwap_shortfall_bps(cls, v, values):
        fill = values.get("fill_price")
        vwap = values.get("period_vwap")
        if fill is not None and vwap not in (None, 0):
            computed = (fill - vwap) / vwap * 10000
            if v is not None and abs(v - computed) > 1e-2:
                raise ValueError(
                    "vwap_shortfall_bps does not match fill_price and period_vwap"
                )
            return round(computed, 4)
        return v

    @validator("execution_duration_seconds")
    def validate_execution_duration(cls, v):
        if v is not None and v < 0:
            raise ValueError("execution_duration_seconds must be non‑negative")
        return v

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
                "order_id": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
                "signal_price": 102.34,
                "expected_price": 102.50,
                "fill_price": 102.55,
                "slippage_bps": 4.88,
                "execution_algo": "TWAP",
                "created_at": "2024-01-15T12:34:56Z",
                "arrival_price": 102.48,
                "is_cost_bps": 6.86,
                "vwap_shortfall_bps": 2.15,
                "period_vwap": 102.52,
                "execution_duration_seconds": 0.75,
            }
        }