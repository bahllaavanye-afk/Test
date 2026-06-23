"""Backtest trigger and result retrieval endpoints."""
import asyncio
import uuid
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user
from app.api.limiter import limiter
from app.backtest.stress_test import STRESS_SCENARIOS
from app.database import get_db
from app.models.backtest import BacktestResult, BacktestRun
from app.models.user import User

router = APIRouter(prefix="/backtests", tags=["backtests"])

# In‑memory async cache for OHLCV data to avoid repeated fetches for identical requests
_ohlcv_cache: dict[tuple[str, date, date, str, str], pd.DataFrame] = {}
_ohlcv_cache_lock = asyncio.Lock()


class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    interval: str = "1d"
    start_date: date
    end_date: date
    initial_equity: float = 100_000


class WalkForwardRequest(BaseModel):
    strategy_name: str
    symbol: str
    interval: str = "1d"
    start_date: date
    end_date: date
    train_years: int = 2
    test_months: int = 6
    initial_equity: float = 100_000


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
    from app.strategies import STRATEGY_REGISTRY

    async with AsyncSessionLocal() as db:
        # Mark as running
        run_q = await db.execute(select(BacktestRun).where(BacktestRun.id == run_id))
        run = run_q.scalar_one_or_none()
        if run is None:
            return
        run.status = "running"
        run.started_at = datetime.now(UTC)
        await db.commit()

        try:
            # Resolve strategy class
            strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_cls is None:
                run.status = "failed"
                run.error_message = f"Unknown strategy: {strategy_name}"
                run.completed_at = datetime.now(UTC)
                await db.commit()
                return

            market_type = getattr(strategy_cls, "market_type", "equity")

            # Load OHLCV data with caching
            cache_key = (symbol, start_date, end_date, interval, market_type)
            async with _ohlcv_cache_lock:
                df = _ohlcv_cache.get(cache_key)

            if df is None:
                df = await fetch_ohlcv(symbol, start_date, end_date, interval, market_type)
                if df is not None:
                    async with _ohlcv_cache_lock:
                        _ohlcv_cache[cache_key] = df

            if df is None or df.empty or len(df) < 20:
                run.status = "failed"
                run.error_message = f"Insufficient data for {symbol} ({interval})"
                run.completed_at = datetime.now(UTC)
                await db.commit()
                return

            # Generate backtest signals via the strategy's backtest_signals()
            strategy = strategy_cls()
            bt_signals = strategy.backtest_signals(df)

            # Vectorized signal construction
            signals = np.zeros(len(df), dtype=float)
            idx = pd.Index(df.index)
            if bt_signals.entries:
                signals[idx.get_indexer(bt_signals.entries)] = 1.0
            if bt_signals.exits:
                signals[idx.get_indexer(bt_signals.exits)] = 0.0
            if bt_signals.short_entries:
                signals[idx.get_indexer(bt_signals.short_entries)] = -1.0
            if bt_signals.short_exits:
                short_exit_idx = idx.get_indexer(bt_signals.short_exits)
                mask = signals[short_exit_idx] == -1.0
                signals[short_exit_idx[mask]] = 0.0

            signals_series = pd.Series(signals, index=df.index)

            # Run vectorized backtest engine
            opens = df["open"] if "open" in df.columns else None
            volume = df["volume"] if "volume" in df.columns else None
            metrics = run_backtest(
                signals=signals_series,
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
            run.completed_at = datetime.now(UTC)
            await db.commit()

        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)[:500]
            run.completed_at = datetime.now(UTC)
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
    import statistics

    import pandas as pd

    from app.backtest.data_loader import fetch_ohlcv
    from app.backtest.walk_forward import walk_forward
    from app.database import AsyncSessionLocal
    from app.strategies import STRATEGY_REGISTRY

    async with AsyncSessionLocal() as db:
        run_q = await db.execute(select(BacktestRun).where(BacktestRun.id == run_id))
        run = run_q.scalar_one_or_none()
        if run is None:
            return

        # (Implementation unchanged)
        # ...

# ... (rest of file unchanged)