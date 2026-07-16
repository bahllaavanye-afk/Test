"""Cross-desk tracking fields on Position/Order (desk consolidation stage 3)."""
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, validator

from app.models.order import Order
from app.models.position import Position

_FIELDS = {"asset_class", "underlying_symbol", "expiry", "strike",
           "option_right", "contract_multiplier"}


def test_position_has_cross_desk_fields():
    assert _FIELDS <= set(Position.__table__.columns.keys())


def test_order_has_cross_desk_fields():
    assert _FIELDS <= set(Order.__table__.columns.keys())


def test_asset_class_defaults_to_equity():
    for model in (Position, Order):
        col = model.__table__.columns["asset_class"]
        assert col.default.arg == "equity"
        assert col.nullable is False


class CrossDeskFields(BaseModel):
    """Pydantic schema representing the cross‑desk fields shared by Position and Order."""

    asset_class: str = Field(
        ...,
        description="Asset class of the instrument.",
        example="equity",
    )
    underlying_symbol: str = Field(
        ...,
        description="Ticker symbol of the underlying security.",
        example="AAPL",
    )
    expiry: Optional[date] = Field(
        None,
        description="Expiration date for options or futures contracts.",
        example="2025-12-31",
    )
    strike: Optional[float] = Field(
        None,
        description="Strike price for options contracts.",
        example=150.0,
    )
    option_right: Optional[str] = Field(
        None,
        description="Option right type; either 'call' or 'put'.",
        example="call",
    )
    contract_multiplier: int = Field(
        ...,
        description="Number of underlying units per contract.",
        example=100,
    )

    @validator("expiry")
    def validate_expiry_future(cls, v: Optional[date]) -> Optional[date]:
        if v is not None and v <= date.today():
            raise ValueError("expiry must be a future date")
        return v

    @validator("strike")
    def validate_strike_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("strike must be a positive number")
        return v

    @validator("option_right")
    def validate_option_right(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in {"call", "put"}:
            raise ValueError("option_right must be either 'call' or 'put'")
        return v.lower() if v is not None else v

    @validator("contract_multiplier")
    def validate_contract_multiplier_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("contract_multiplier must be a positive integer")
        return v


__all__ = [
    "CrossDeskFields",
    "_FIELDS",
    "test_position_has_cross_desk_fields",
    "test_order_has_cross_desk_fields",
    "test_asset_class_defaults_to_equity",
]