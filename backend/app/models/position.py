import uuid
from datetime import date, datetime
from typing import Optional, Literal

from sqlalchemy import String, ForeignKey, Numeric, DateTime, Date, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pydantic import BaseModel, Field, validator

from app.database import Base


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("account_id", "symbol"),)

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.id", ondelete="CASCADE")
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)   # long|short
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    avg_cost: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    current_price: Mapped[float | None] = mapped_column(Numeric(18, 8))
    unrealized_pnl: Mapped[float | None] = mapped_column(Numeric(18, 8))
    # Cross-desk tracking — one position shape for every desk (equity/crypto/option/...).
    asset_class: Mapped[str] = mapped_column(
        String(16), nullable=False, default="equity"
    )
    underlying_symbol: Mapped[str | None] = mapped_column(String(32))  # options: the underlying
    expiry: Mapped[date | None] = mapped_column(Date)                  # options/futures
    strike: Mapped[float | None] = mapped_column(Numeric(18, 8))       # options
    option_right: Mapped[str | None] = mapped_column(String(4))        # call|put
    contract_multiplier: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1
    )
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    account: Mapped["Account"] = relationship(
        "Account", back_populates="positions"
    )


class PositionSchema(BaseModel):
    """
    Pydantic schema for Position model with field descriptions,
    examples, and validation.
    """

    id: str = Field(
        ...,
        description="Unique identifier for the position.",
        example="550e8400-e29b-41d4-a716-446655440000",
    )
    account_id: str = Field(
        ...,
        description="Reference to the owning account.",
        example="acc_12345",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol of the asset.",
        example="AAPL",
        max_length=32,
    )
    side: Literal["long", "short"] = Field(
        ...,
        description="Position side – long or short.",
        example="long",
    )
    quantity: float = Field(
        ...,
        description="Number of units held.",
        example=100.0,
        ge=0,
    )
    avg_cost: float = Field(
        ...,
        description="Average cost basis per unit.",
        example=150.0,
        ge=0,
    )
    current_price: Optional[float] = Field(
        None,
        description="Latest market price of the underlying.",
        example=155.0,
        ge=0,
    )
    unrealized_pnl: Optional[float] = Field(
        None,
        description="Unrealized profit and loss.",
        example=500.0,
    )
    asset_class: str = Field(
        "equity",
        description="Asset class classification (e.g., equity, crypto).",
        example="equity",
        max_length=16,
    )
    underlying_symbol: Optional[str] = Field(
        None,
        description="Underlying symbol for derivatives.",
        example="SPY",
        max_length=32,
    )
    expiry: Optional[date] = Field(
        None,
        description="Expiration date for options/futures.",
        example="2025-12-31",
    )
    strike: Optional[float] = Field(
        None,
        description="Strike price for options.",
        example=200.0,
        ge=0,
    )
    option_right: Optional[Literal["call", "put"]] = Field(
        None,
        description="Option right – call or put.",
        example="call",
    )
    contract_multiplier: int = Field(
        1,
        description="Contract multiplier (e.g., 100 for standard options).",
        example=1,
        gt=0,
    )
    opened_at: datetime = Field(
        ...,
        description="Timestamp when the position was opened.",
        example="2024-01-01T09:30:00Z",
    )
    updated_at: datetime = Field(
        ...,
        description="Timestamp of the last update to the position.",
        example="2024-01-02T10:45:00Z",
    )

    @validator("expiry")
    def validate_expiry_not_past(cls, v: Optional[date]) -> Optional[date]:
        if v is not None and v < date.today():
            raise ValueError("expiry date cannot be in the past")
        return v

    @validator("option_right")
    def validate_option_right(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {"call", "put"}:
            raise ValueError("option_right must be 'call' or 'put'")
        return v

    class Config:
        orm_mode = True
        anystr_strip_whitespace = True
        title = "Position"
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "account_id": "acc_12345",
                "symbol": "AAPL",
                "side": "long",
                "quantity": 100.0,
                "avg_cost": 150.0,
                "current_price": 155.0,
                "unrealized_pnl": 500.0,
                "asset_class": "equity",
                "underlying_symbol": None,
                "expiry": None,
                "strike": None,
                "option_right": None,
                "contract_multiplier": 1,
                "opened_at": "2024-01-01T09:30:00Z",
                "updated_at": "2024-01-02T10:45:00Z",
            }
        }