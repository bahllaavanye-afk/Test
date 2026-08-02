import uuid
from datetime import datetime, date
from typing import List, Optional, Dict, Any

from sqlalchemy import String, ForeignKey, Numeric, DateTime, Date, Integer, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pydantic import BaseModel, Field, validator, root_validator

from app.database import Base


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    params: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(
        String(16), default="queued"
    )  # queued|running|done|failed
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    result: Mapped["BacktestResult | None"] = relationship(
        "BacktestResult", back_populates="run", uselist=False
    )


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    run_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("backtest_runs.id", ondelete="CASCADE"),
        unique=True,
    )
    total_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    annualized_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4))
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    sortino_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    calmar_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    max_drawdown: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    win_rate: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    profit_factor: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    total_trades: Mapped[Optional[int]] = mapped_column(Integer)
    equity_curve: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)  # [{ts, value}, ...]
    trades_log: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)  # [{entry, exit, pnl}, ...]

    run: Mapped["BacktestRun"] = relationship(
        "BacktestRun", back_populates="result"
    )


# Pydantic schemas -----------------------------------------------------------


class BacktestRunSchema(BaseModel):
    """Schema representing a backtest run request/response."""

    id: str = Field(
        ..., description="Unique identifier of the backtest run", example="c1a2b3d4-5678-90ab-cdef-1234567890ab"
    )
    user_id: str = Field(..., description="Identifier of the user who initiated the run", example="user_42")
    strategy_name: str = Field(..., description="Name of the strategy to backtest", example="mean_rev_20_2")
    symbol: str = Field(..., description="Ticker symbol under test", example="AAPL")
    interval: str = Field(..., description="Timeframe interval (e.g., 1m, 5m, 1h)", example="1h")
    start_date: date = Field(..., description="Inclusive start date for the backtest", example="2023-01-01")
    end_date: date = Field(..., description="Inclusive end date for the backtest", example="2023-12-31")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy‑specific parameters",
        example={"window": 20, "threshold": 2.0},
    )
    status: str = Field(
        default="queued",
        description="Current execution status",
        example="running",
    )
    started_at: Optional[datetime] = Field(
        None, description="Timestamp when execution started", example="2024-01-15T09:30:00Z"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Timestamp when execution finished", example="2024-01-15T10:45:00Z"
    )
    error_message: Optional[str] = Field(
        None, description="Error details if the run failed", example="Division by zero"
    )
    created_at: datetime = Field(
        ..., description="Record creation timestamp", example="2024-01-15T09:00:00Z"
    )
    result: Optional["BacktestResultSchema"] = Field(
        None, description="Associated backtest result when available"
    )

    class Config:
        orm_mode = True

    @validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"queued", "running", "done", "failed"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v

    @root_validator
    def check_dates(cls, values):
        start = values.get("start_date")
        end = values.get("end_date")
        if start and end and start > end:
            raise ValueError("start_date must be on or before end_date")
        return values


class BacktestResultSchema(BaseModel):
    """Schema representing the outcome metrics of a backtest."""

    id: str = Field(..., description="Unique identifier of the result record", example="d4e5f6a7-8910-11ab-cdef-1234567890ab")
    run_id: str = Field(..., description="Foreign key linking to the backtest run", example="c1a2b3d4-5678-90ab-cdef-1234567890ab")
    total_return: Optional[float] = Field(
        None, description="Cumulative return over the period", example=0.1523
    )
    annualized_return: Optional[float] = Field(
        None, description="Annualized return", example=0.1845
    )
    sharpe_ratio: Optional[float] = Field(
        None, description="Sharpe ratio", example=1.35
    )
    sortino_ratio: Optional[float] = Field(
        None, description="Sortino ratio", example=1.80
    )
    calmar_ratio: Optional[float] = Field(
        None, description="Calmar ratio", example=0.75
    )
    max_drawdown: Optional[float] = Field(
        None, description="Maximum drawdown (as a decimal)", example=0.12
    )
    win_rate: Optional[float] = Field(
        None,
        description="Proportion of winning trades (0‑1)",
        example=0.56,
        ge=0.0,
        le=1.0,
    )
    profit_factor: Optional[float] = Field(
        None, description="Profit factor", example=1.45
    )
    total_trades: Optional[int] = Field(
        None, description="Number of trades executed", example=124
    )
    equity_curve: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Time‑series of equity values",
        example=[{"ts": "2023-01-01T00:00:00Z", "value": 100000}, {"ts": "2023-01-02T00:00:00Z", "value": 101200}],
    )
    trades_log: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Log of individual trades",
        example=[{"entry": "2023-01-01T09:30:00Z", "exit": "2023-01-01T10:00:00Z", "pnl": 250.0}],
    )

    class Config:
        orm_mode = True

    @validator("win_rate")
    def validate_win_rate(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("win_rate must be between 0 and 1")
        return v

# Resolve forward references
BacktestRunSchema.update_forward_refs()
BacktestResultSchema.update_forward_refs()