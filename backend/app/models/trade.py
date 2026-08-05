import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, ForeignKey, Numeric, DateTime, Integer, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column
from pydantic import BaseModel, Field, validator

from app.database import Base


class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_account_closed", "account_id", "closed_at"),
        Index("ix_trades_strategy_closed", "strategy_id", "closed_at"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id: Mapped[str] = mapped_column(String, ForeignKey("accounts.id", ondelete="CASCADE"), index=True)
    strategy_id: Mapped[str | None] = mapped_column(String, ForeignKey("strategies.id", ondelete="SET NULL"), index=True)
    # Denormalized for fast attribution queries (avoids JOIN to strategies table)
    strategy_name: Mapped[str | None] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    exit_price: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    quantity: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Numeric(18, 8), nullable=False)
    fees: Mapped[float] = mapped_column(Numeric(18, 8), default=0)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    hold_seconds: Mapped[int | None] = mapped_column(Integer)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict)


class TradeSchema(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier for the trade.",
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    account_id: str = Field(
        ...,
        description="Foreign key reference to the account.",
        example="acc_001",
    )
    strategy_id: Optional[str] = Field(
        None,
        description="Foreign key reference to the strategy.",
        example="strat_01",
    )
    strategy_name: Optional[str] = Field(
        None,
        description="Denormalized strategy name for quick attribution.",
        example="Mean Reversion 20",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol of the traded instrument.",
        example="AAPL",
    )
    side: str = Field(
        ...,
        description="Trade side, either 'buy' or 'sell'.",
        example="buy",
    )
    entry_price: float = Field(
        ...,
        description="Price at which the position was opened.",
        example=150.25,
    )
    exit_price: float = Field(
        ...,
        description="Price at which the position was closed.",
        example=152.00,
    )
    quantity: float = Field(
        ...,
        description="Number of units traded.",
        example=100,
    )
    realized_pnl: float = Field(
        ...,
        description="Realized profit and loss from the trade.",
        example=175.0,
    )
    fees: float = Field(
        0,
        description="Total fees paid for the trade.",
        example=2.5,
    )
    opened_at: datetime = Field(
        ...,
        description="Timestamp when the trade was opened.",
        example="2023-01-01T09:30:00Z",
    )
    closed_at: datetime = Field(
        ...,
        description="Timestamp when the trade was closed.",
        example="2023-01-01T10:15:00Z",
    )
    hold_seconds: Optional[int] = Field(
        None,
        description="Duration the trade was held in seconds.",
        example=2700,
    )
    raw_payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original raw payload data from execution source.",
        example={"order_id": "ord123"},
    )

    @validator("side")
    def validate_side(cls, v: str) -> str:
        lowered = v.lower()
        if lowered not in {"buy", "sell"}:
            raise ValueError("side must be either 'buy' or 'sell'")
        return lowered

    @validator("quantity")
    def validate_quantity(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("quantity must be positive")
        return v

    @validator("entry_price", "exit_price")
    def validate_price(cls, v: float, field) -> float:
        if v < 0:
            raise ValueError(f"{field.name} must be non‑negative")
        return v

    @validator("fees")
    def validate_fees(cls, v: float) -> float:
        if v < 0:
            raise ValueError("fees must be non‑negative")
        return v

    @validator("hold_seconds")
    def validate_hold_seconds(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("hold_seconds must be non‑negative")
        return v

    class Config:
        orm_mode = True