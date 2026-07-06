"""Backtest trigger and result retrieval endpoints."""
from datetime import date, datetime, timezone
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user
from app.api.limiter import limiter
from app.backtest.stress_test import STRESS_SCENARIOS
from app.database import get_db
from app.models.backtest import BacktestRun, BacktestResult
from app.models.user import User
from app.strategies import STRATEGY_REGISTRY

router = APIRouter(prefix="/backtests", tags=["backtests"])


class BacktestRequest(BaseModel):
    strategy_name: str = Field(
        ...,
        description="Identifier of the strategy to back‑test (must exist in the strategy registry).",
        example="mean_rev_20_1.5",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol or asset identifier for which the back‑test is executed.",
        example="SPY",
    )
    interval: str = Field(
        "1d",
        description="Data interval (e.g., '1d', '5m', '1h'). Must be compatible with the data provider.",
        example="1d",
    )
    start_date: date = Field(
        ...,
        description="Inclusive start date for the back‑test period.",
        example="2023-01-01",
    )
    end_date: date = Field(
        ...,
        description="Inclusive end date for the back‑test period. Must be later than start_date.",
        example="2023-12-31",
    )
    initial_equity: float = Field(
        100_000,
        ge=0,
        description="Starting capital for the simulation.",
        example=100_000,
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("initial_equity")
    @classmethod
    def equity_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("initial_equity must be greater than zero")
        return v

    @model_validator(mode="after")
    def check_dates(cls, values):
        if values.start_date >= values.end_date:
            raise ValueError("start_date must be earlier than end_date")
        return values


class WalkForwardRequest(BaseModel):
    strategy_name: str = Field(
        ...,
        description="Identifier of the strategy to evaluate with walk‑forward validation.",
        example="mean_rev_20_2",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol or asset identifier for which the walk‑forward test is executed.",
        example="AAPL",
    )
    interval: str = Field(
        "1d",
        description="Data interval (e.g., '1d', '5m', '1h').",
        example="1d",
    )
    start_date: date = Field(
        ...,
        description="Inclusive start date for the training period.",
        example="2020-01-01",
    )
    end_date: date = Field(
        ...,
        description="Inclusive end date for the testing period. Must be later than start_date.",
        example="2023-12-31",
    )
    train_years: int = Field(
        2,
        ge=1,
        description="Number of years to use for each training window.",
        example=2,
    )
    test_months: int = Field(
        6,
        ge=1,
        description="Number of months for each testing window.",
        example=6,
    )
    initial_equity: float = Field(
        100_000,
        ge=0,
        description="Starting capital for each walk‑forward iteration.",
        example=100_000,
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("initial_equity")
    @classmethod
    def equity_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("initial_equity must be greater than zero")
        return v

    @field_validator("train_years", "test_months")
    @classmethod
    def positive_ints(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Values must be at least 1")
        return v

    @model_validator(mode="after")
    def check_dates(cls, values):
        if values.start_date >= values.end_date:
            raise ValueError("start_date must be earlier than end_date")
        return values


class BacktestOut(BaseModel):
    id: str = Field(..., description="Unique identifier of the back‑test run.", example="b1a2c3d4")
    strategy_name: str = Field(..., description="Name of the strategy that was back‑tested.", example="mean_rev_20_1.5")
    symbol: str = Field(..., description="Asset symbol used in the back‑test.", example="SPY")
    interval: str = Field(..., description="Data interval used for the back‑test.", example="1d")
    status: str = Field(..., description="Current status of the back‑test run (e.g., pending, running, done, failed).", example="done")
    sharpe: float | None = Field(None, description="Sharpe ratio of the back‑test results.", example=1.52)
    sortino: float | None = Field(None, description="Sortino ratio of the back‑test results.", example=2.01)
    calmar: float | None = Field(None, description="Calmar ratio of the back‑test results.", example=0.85)
    max_drawdown: float | None = Field(None, description="Maximum drawdown observed during the back‑test.", example=-0.15)
    total_return: float | None = Field(None, description="Total return over the back‑test period.", example=0.23)
    annualized_return: float | None = Field(None, description="Annualized return derived from total return.", example=0.12)
    win_rate: float | None = Field(None, description="Proportion of winning trades.", example=0.57)
    profit_factor: float | None = Field(None, description="Profit factor (gross profit / gross loss).", example=1.34)
    total_trades: int | None = Field(None, description="Number of trades executed during the back‑test.", example=124)
    equity_curve: list | None = Field(
        None,
        description="Sample of the equity curve (list of equity values). Truncated to first 500 points.",
        example=[100000, 101200, 100850],
    )
    error_message: str | None = Field(None, description="Error message if the back‑test failed.", example="Insufficient data for symbol.")
    created_at: datetime = Field(..., description="Timestamp when the back‑test request was created.", example="2024-01-01T12:00:00Z")
    started_at: datetime | None = Field(None, description="Timestamp when the back‑test began execution.", example="2024-01-01T12:05:00Z")
    completed_at: datetime | None = Field(None, description="Timestamp when the back‑test finished.", example="2024-01-01T12:30:00Z")

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_run(cls, run) -> "BacktestOut":
        result = run.result
        return cls(
            id=run.id,
            strategy_name=run.strategy_name,
            symbol=run.symbol,
            interval=run.interval,
            status=run.status,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
            error_message=run.error_message,
            sharpe=float(result.sharpe_ratio) if result and result.sharpe_ratio is not None else None,
            sortino=float(result.sortino_ratio) if result and result.sortino_ratio is not None else None,
            calmar=float(result.calmar_ratio) if result and result.calmar_ratio is not None else None,
            max_drawdown=float(result.max_drawdown) if result and result.max_drawdown is not None else None,
            total_return=float(result.total_return) if result and result.total_return is not None else None,
            annualized_return=float(result.annualized_return) if result and result.annualized_return is not None else None,
            win_rate=float(result.win_rate) if result and result.win_rate is not None else None,
            profit_factor=float(result.profit_factor) if result and result.profit_factor is not None else None,
            total_trades=result.total_trades if result else None,
            equity_curve=result.equity_curve if result else None,
        )


async def _run_backtest_task(
    run_id: str,
    strategy_name: str,
    symbol: str,
    interval: str,
    start_date: date,
    end_date: date,
    initial_equity: float,
) -> None:
    """Background task: fetch OHLCV, run strategy.backtest_signals(), pass to engine."""
    from app.backtest.data_loader import fetch_ohlcv
    from app.backtest.engine import run_backtest
    from app.database import AsyncSessionLocal
    import pandas as pd

    async with AsyncSessionLocal() as db:
        # Mark as running
        run_q = await db.execute(select(BacktestRun).where(BacktestRun.id == run_id))
        run = run_q.scalar_one_or_none()
        if run is None:
            return
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        await db.commit()

        try:
            # Resolve strategy class
            strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_cls is None:
                run.status = "failed"
                run.error_message = f"Unknown strategy: {strategy_name}"
                run.completed_at = datetime.now(timezone.utc)
                await db.commit()
                return

            market_type = getattr(strategy_cls, "market_type", "equity")

            # Load OHLCV data via yfinance (free, no API key)
            df = await fetch_ohlcv(symbol, start_date, end_date, interval, market_type)
            if df is None or df.empty or len(df) < 20:
                run.status = "failed"
                run.error_message = f"Insufficient data for {symbol} ({interval})"
                run.completed_at = datetime.now(timezone.utc)
                await db.commit()
                return

            # Generate backtest signals via the strategy's backtest_signals()
            strategy = strategy_cls()
            bt_signals = strategy.backtest_signals(df)

            # Convert BacktestSignals → numeric signal series (-1/0/+1)
            signals = pd.Series(0.0, index=df.index)
            signals[bt_signals.entries] = 1.0
            signals[bt_signals.exits] = 0.0
            if bt_signals.short_entries is not None:
                signals[bt_signals.short_entries] = -1.0
            if bt_signals.short_exits is not None:
                signals[bt_signals.short_exits & (signals == -1.0)] = 0.0

            # Run vectorized backtest engine
            opens = df["open"] if "open" in df.columns else None
            volume = df["volume"] if "volume" in df.columns else None
            metrics = run_backtest(
                signals=signals,
                prices=df["close"],
                opens=opens,
                volume=volume,
                initial_equity=initial_equity,
            )

            # Persist result to DB
            result = BacktestResult(
                id=str(uuid.uuid4()),
                run_id=run_id,
                total_return=round(metrics.total_return, 6),
                annualized_return=round(metrics.annualized_return, 6),
                sharpe_ratio=round(metrics.sharpe, 4),
                sortino_ratio=round(metrics.sortino, 4),
                calmar_ratio=round(metrics.calmar, 4),
                max_drawdown=round(metrics.max_drawdown, 4),
                win_rate=round(metrics.win_rate, 4),
                profit_factor=round(metrics.profit_factor, 4),
                total_trades=metrics.num_trades,
                equity_curve=metrics.equity_curve[:500],  # cap payload size
            )
            db.add(result)

            run.status = "done"
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()

        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)[:500]
            run.completed_at = datetime.now(timezone.utc)
            try:
                await db.commit()
            except Exception:
                pass


async def _run_walk_forward_task(
    run_id: str,
    strategy_name: str,
    symbol: str,
    interval: str,
    start_date: date,
    end_date: date,
    train_years: int,
    test_months: int,
    initial_equity: float,
) -> None:
    """Background task: walk-forward validation using strategy.backtest_signals()."""
    from app.backtest.data_loader import fetch_ohlcv
    from app.backtest.walk_forward import walk_forward
    from app.database import AsyncSessionLocal
    import pandas as pd
    import statistics

    async with AsyncSessionLocal() as db:
        run_q = await db.execute(select(BacktestRun).where(BacktestRun.id == run_id))
        run = run_q.scalar_one_or_none()
        # Remaining implementation omitted for brevity.