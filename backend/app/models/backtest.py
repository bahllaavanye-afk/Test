import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from sqlalchemy import Date, DateTime, ForeignKey, Integer, JSON, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from pydantic import BaseModel, Field, root_validator, validator


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    strategy_name: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    result: Mapped["BacktestResult | None"] = relationship(
        "BacktestResult", back_populates="run", uselist=False
    )


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("backtest_runs.id", ondelete="CASCADE"),
        unique=True,
    )
    total_return: Mapped[float | None] = mapped_column(Numeric(10, 4))
    annualized_return: Mapped[float | None] = mapped_column(Numeric(10, 4))
    sharpe_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))
    sortino_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))
    calmar_ratio: Mapped[float | None] = mapped_column(Numeric(8, 4))
    max_drawdown: Mapped[float | None] = mapped_column(Numeric(8, 4))
    win_rate: Mapped[float | None] = mapped_column(Numeric(6, 4))
    profit_factor: Mapped[float | None] = mapped_column(Numeric(8, 4))
    total_trades: Mapped[int | None] = mapped_column(Integer)
    equity_curve: Mapped[list | None] = mapped_column(JSON)  # [{ts, value}, ...]
    trades_log: Mapped[list | None] = mapped_column(JSON)  # [{entry, exit, pnl}, ...]

    run: Mapped["BacktestRun"] = relationship("BacktestRun", back_populates="result")


class BacktestRunSchema(BaseModel):
    """Pydantic schema for BacktestRun model."""

    id: str = Field(
        ...,
        description="Unique identifier for the backtest run.",
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    user_id: str = Field(
        ...,
        description="Reference to the user who initiated the run.",
        example="user_42",
    )
    strategy_name: str = Field(
        ...,
        description="Name of the strategy used for the backtest.",
        example="mean_rev_20_2",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol for the backtest.",
        example="AAPL",
    )
    interval: str = Field(
        ...,
        description="Timeframe interval (e.g., 1h, 1d).",
        example="1d",
    )
    start_date: date = Field(
        ...,
        description="Start date of the backtest period.",
        example="2023-01-01",
    )
    end_date: date = Field(
        ...,
        description="End date of the backtest period.",
        example="2023-12-31",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy parameters supplied to the backtest.",
        example={"window": 20, "threshold": 0.02},
    )
    status: str = Field(
        default="queued",
        description="Current status of the backtest run.",
        example="running",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Timestamp when execution started.",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when execution finished.",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if the run failed.",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the record was created.",
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
    def check_date_order(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        start = values.get("start_date")
        end = values.get("end_date")
        if start and end and end < start:
            raise ValueError("end_date must be after start_date")
        return values


class BacktestResultSchema(BaseModel):
    """Pydantic schema for BacktestResult model."""

    id: str = Field(
        ...,
        description="Unique identifier for the backtest result.",
        example="223e4567-e89b-12d3-a456-426614174111",
    )
    run_id: str = Field(
        ...,
        description="Foreign key linking to the associated BacktestRun.",
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    total_return: Optional[float] = Field(
        None,
        description="Total return of the strategy over the backtest period.",
        example=0.12,
    )
    annualized_return: Optional[float] = Field(
        None,
        description="Annualized return of the strategy.",
        example=0.15,
    )
    sharpe_ratio: Optional[float] = Field(
        None,
        description="Sharpe ratio of the strategy.",
        example=1.2,
    )
    sortino_ratio: Optional[float] = Field(
        None,
        description="Sortino ratio of the strategy.",
        example=1.5,
    )
    calmar_ratio: Optional[float] = Field(
        None,
        description="Calmar ratio of the strategy.",
        example=0.8,
    )
    max_drawdown: Optional[float] = Field(
        None,
        description="Maximum drawdown observed during the backtest.",
        example=-0.2,
    )
    win_rate: Optional[float] = Field(
        None,
        description="Proportion of winning trades.",
        example=0.55,
    )
    profit_factor: Optional[float] = Field(
        None,
        description="Profit factor (gross profit / gross loss).",
        example=1.3,
    )
    total_trades: Optional[int] = Field(
        None,
        description="Total number of trades executed.",
        example=150,
    )
    equity_curve: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Equity curve data points (timestamp and portfolio value).",
        example=[{"ts": "2023-01-01", "value": 1000.0}],
    )
    trades_log: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Log of individual trades with entry, exit, and P&L.",
        example=[{"entry": "2023-01-02", "exit": "2023-01-05", "pnl": 10.0}],
    )

    class Config:
        orm_mode = True

    @validator("total_trades")
    def validate_total_trades(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("total_trades must be non‑negative")
        return v

    @validator("win_rate")
    def validate_win_rate(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("win_rate must be between 0 and 1")
        return v