"""Backtest trigger and result retrieval endpoints."""
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.database import get_db
from app.api.deps import get_current_user
from app.api.limiter import limiter
from app.models.backtest import BacktestRun, BacktestResult
from app.models.user import User
from app.backtest.stress_test import STRESS_SCENARIOS
from pydantic import BaseModel, ConfigDict
from datetime import date, datetime, timezone
import uuid

# Constants
DEFAULT_INTERVAL = "1d"
DEFAULT_INITIAL_EQUITY = 100_000
DEFAULT_TRAIN_YEARS = 2
DEFAULT_TEST_MONTHS = 6

STATUS_RUNNING = "running"
STATUS_FAILED = "failed"
STATUS_DONE = "done"

MIN_DATA_LENGTH = 20
EQUITY_CURVE_MAX_LENGTH = 500
ERROR_MESSAGE_MAX_LENGTH = 500

ROUND_TOTAL_RETURN_PRECISION = 6
ROUND_OTHER_PRECISION = 4

router = APIRouter(prefix="/backtests", tags=["backtests"])


class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    interval: str = DEFAULT_INTERVAL
    start_date: date
    end_date: date
    initial_equity: float = DEFAULT_INITIAL_EQUITY


class WalkForwardRequest(BaseModel):
    strategy_name: str
    symbol: str
    interval: str = DEFAULT_INTERVAL
    start_date: date
    end_date: date
    train_years: int = DEFAULT_TRAIN_YEARS
    test_months: int = DEFAULT_TEST_MONTHS
    initial_equity: float = DEFAULT_INITIAL_EQUITY


class BacktestOut(BaseModel):
    id: str
    strategy_name: str
    symbol: str
    interval: str
    status: str
    sharpe: float | None = None
    sortino: float | None = None
    calmar: float | None = None
    max_drawdown: float | None = None
    total_return: float | None = None
    annualized_return: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    total_trades: int | None = None
    equity_curve: list | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

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


async def _run_backtest_task(run_id: str, strategy_name: str, symbol: str,
                              interval: str, start_date: date, end_date: date,
                              initial_equity: float) -> None:
    """Background task: fetch OHLCV, run strategy.backtest_signals(), pass to engine."""
    from app.database import AsyncSessionLocal
    from app.backtest.data_loader import fetch_ohlcv
    from app.backtest.engine import run_backtest
    from app.strategies import STRATEGY_REGISTRY
    import pandas as pd

    async with AsyncSessionLocal() as db:
        # Mark as running
        run_q = await db.execute(select(BacktestRun).where(BacktestRun.id == run_id))
        run = run_q.scalar_one_or_none()
        if run is None:
            return
        run.status = STATUS_RUNNING
        run.started_at = datetime.now(timezone.utc)
        await db.commit()

        try:
            # Resolve strategy class
            strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_cls is None:
                run.status = STATUS_FAILED
                run.error_message = f"Unknown strategy: {strategy_name}"
                run.completed_at = datetime.now(timezone.utc)
                await db.commit()
                return

            market_type = getattr(strategy_cls, "market_type", "equity")

            # Load OHLCV data via yfinance (free, no API key)
            df = await fetch_ohlcv(symbol, start_date, end_date, interval, market_type)
            if df is None or df.empty or len(df) < MIN_DATA_LENGTH:
                run.status = STATUS_FAILED
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
                total_return=round(metrics.total_return, ROUND_TOTAL_RETURN_PRECISION),
                annualized_return=round(metrics.annualized_return, ROUND_TOTAL_RETURN_PRECISION),
                sharpe_ratio=round(metrics.sharpe, ROUND_OTHER_PRECISION),
                sortino_ratio=round(metrics.sortino, ROUND_OTHER_PRECISION),
                calmar_ratio=round(metrics.calmar, ROUND_OTHER_PRECISION),
                max_drawdown=round(metrics.max_drawdown, ROUND_OTHER_PRECISION),
                win_rate=round(metrics.win_rate, ROUND_OTHER_PRECISION),
                profit_factor=round(metrics.profit_factor, ROUND_OTHER_PRECISION),
                total_trades=metrics.num_trades,
                equity_curve=metrics.equity_curve[:EQUITY_CURVE_MAX_LENGTH],
            )
            db.add(result)

            run.status = STATUS_DONE
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()

        except Exception as exc:
            run.status = STATUS_FAILED
            run.error_message = str(exc)[:ERROR_MESSAGE_MAX_LENGTH]
            run.completed_at = datetime.now(timezone.utc)
            try:
                await db.commit()
            except Exception:
                pass


async def _run_walk_forward_task(run_id: str, strategy_name: str, symbol: str,
                                  interval: str, start_date: date, end_date: date,
                                  train_years: int, test_months: int,
                                  initial_equity: float) -> None:
    """Background task: walk-forward validation using strategy.backtest_signals()."""
    from app.database import AsyncSessionLocal
    from app.backtest.data_loader import fetch_ohlcv
    from app.backtest.walk_forward import walk_forward
    from app.strategies import STRATEGY_REGISTRY
    import pandas as pd
    import statistics

    async with AsyncSessionLocal() as db:
        run_q = await db.execute(select(BacktestRun).where(BacktestRun.id == run_id))
        run = run_q.scalar_one_or_none()
        # Remaining implementation omitted for brevity
        # ... (truncated for brevity)