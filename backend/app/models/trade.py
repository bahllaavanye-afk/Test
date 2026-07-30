import uuid
from datetime import datetime
from typing import Optional, Dict

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

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="Primary key for the trade record",
    )
    account_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("accounts.id", ondelete="CASCADE"),
        index=True,
        comment="Reference to the account that executed the trade",
    )
    strategy_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("strategies.id", ondelete="SET NULL"),
        index=True,
        comment="Reference to the strategy that generated the trade, if any",
    )
    # Denormalized for fast attribution queries (avoids JOIN to strategies table)
    strategy_name: Mapped[Optional[str]] = mapped_column(
        String(128),
        index=True,
        comment="Cached name of the strategy for quick look‑ups",
    )
    symbol: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Ticker symbol of the traded instrument",
    )
    side: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        comment="Trade side – either 'buy' or 'sell'",
    )
    entry_price: Mapped[float] = mapped_column(
        Numeric(18, 8),
        nullable=False,
        comment="Price at which the position was opened",
    )
    exit_price: Mapped[float] = mapped_column(
        Numeric(18, 8),
        nullable=False,
        comment="Price at which the position was closed",
    )
    quantity: Mapped[float] = mapped_column(
        Numeric(18, 8),
        nullable=False,
        comment="Number of units/contracts traded",
    )
    realized_pnl: Mapped[float] = mapped_column(
        Numeric(18, 8),
        nullable=False,
        comment="Profit or loss realized on the trade",
    )
    fees: Mapped[float] = mapped_column(
        Numeric(18, 8),
        default=0,
        comment="Total fees incurred for the trade",
    )
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Timestamp when the trade was opened",
    )
    closed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Timestamp when the trade was closed",
    )
    hold_seconds: Mapped[Optional[int]] = mapped_column(
        Integer,
        comment="Duration of the trade in seconds",
    )
    raw_payload: Mapped[Dict] = mapped_column(
        JSON,
        default=dict,
        comment="Original payload received from the execution venue",
    )


class TradeSchema(BaseModel):
    """
    Pydantic schema for Trade model used in API responses and validation.
    """

    id: str = Field(
        ...,
        description="Unique identifier for the trade",
        example="a3f5c9e2-4d6b-4c8a-9f2e-1b2c3d4e5f6a",
    )
    account_id: str = Field(
        ...,
        description="Identifier of the account that executed the trade",
        example="acc_12345",
    )
    strategy_id: Optional[str] = Field(
        None,
        description="Identifier of the strategy that generated the trade",
        example="strat_01",
    )
    strategy_name: Optional[str] = Field(
        None,
        description="Human‑readable name of the strategy",
        example="Mean Reversion 20‑2",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol of the traded instrument",
        example="AAPL",
    )
    side: str = Field(
        ...,
        description="Side of the trade – either 'buy' or 'sell'",
        example="buy",
    )
    entry_price: float = Field(
        ...,
        description="Price at which the position was opened",
        example=150.25,
    )
    exit_price: float = Field(
        ...,
        description="Price at which the position was closed",
        example=152.80,
    )
    quantity: float = Field(
        ...,
        description="Number of units/contracts traded",
        example=100.0,
    )
    realized_pnl: float = Field(
        ...,
        description="Profit or loss realized on the trade",
        example=250.0,
    )
    fees: float = Field(
        0.0,
        description="Total fees incurred for the trade",
        example=2.5,
    )
    opened_at: datetime = Field(
        ...,
        description="Timestamp when the trade was opened (UTC)",
        example="2024-01-15T13:45:00Z",
    )
    closed_at: datetime = Field(
        ...,
        description="Timestamp when the trade was closed (UTC)",
        example="2024-01-15T14:10:00Z",
    )
    hold_seconds: Optional[int] = Field(
        None,
        description="Duration of the trade in seconds",
        example=1500,
    )
    raw_payload: Dict = Field(
        default_factory=dict,
        description="Original execution payload from the broker or exchange",
        example={"order_id": "ord_9876", "exchange": "NASDAQ"},
    )

    @validator("side")
    def validate_side(cls, v: str) -> str:
        allowed = {"buy", "sell"}
        if v.lower() not in allowed:
            raise ValueError(f"side must be one of {allowed}")
        return v.lower()

    @validator("entry_price", "exit_price", "quantity")
    def validate_positive_numbers(cls, v: float, field) -> float:
        if v <= 0:
            raise ValueError(f"{field.name} must be a positive number")
        return v

    @validator("hold_seconds")
    def validate_hold_seconds(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("hold_seconds must be non‑negative")
        return v

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": "a3f5c9e2-4d6b-4c8a-9f2e-1b2c3d4e5f6a",
                "account_id": "acc_12345",
                "strategy_id": "strat_01",
                "strategy_name": "Mean Reversion 20‑2",
                "symbol": "AAPL",
                "side": "buy",
                "entry_price": 150.25,
                "exit_price": 152.80,
                "quantity": 100.0,
                "realized_pnl": 250.0,
                "fees": 2.5,
                "opened_at": "2024-01-15T13:45:00Z",
                "closed_at": "2024-01-15T14:10:00Z",
                "hold_seconds": 1500,
                "raw_payload": {"order_id": "ord_9876", "exchange": "NASDAQ"},
            }
        }