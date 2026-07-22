"""
Backtest worker — polls for queued BacktestRun rows every 30 s and executes them.

Runs as a background asyncio task started from main.py lifespan.
Uses yfinance for free OHLCV data — no broker keys required.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any

import pandas as pd
from pydantic import BaseModel, Field, validator

from sqlalchemy import select
from app.utils.logging import logger


class BacktestRunSchema(BaseModel):
    """
    Schema representing the input parameters for a backtest run.
    Used for validation, documentation, and IDE assistance.
    """

    id: str = Field(
        ...,
        description="UUID of the backtest run.",
        example="123e4567-e89b-12d3-a456-426614174000",
    )
    symbol: str = Field(
        ...,
        description="Ticker symbol to backtest.",
        example="AAPL",
    )
    start_date: datetime = Field(
        ...,
        description="Start date of the backtest period (UTC).",
        example="2022-01-01T00:00:00Z",
    )
    end_date: datetime = Field(
        ...,
        description="End date of the backtest period (UTC).",
        example="2022-12-31T00:00:00Z",
    )
    interval: str = Field(
        ...,
        description="Data interval, e.g., '1d', '1h'.",
        example="1d",
    )
    strategy_name: str = Field(
        ...,
        description="Name of the strategy registered in STRATEGY_REGISTRY.",
        example="mean_rev_20_2",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional strategy‑specific parameters.",
        example={"initial_equity": 100_000.0},
    )

    @validator("end_date")
    def check_date_order(cls, v: datetime, values: dict) -> datetime:
        """Ensure that end_date is later than start_date."""
        start = values.get("start_date")
        if start is not None and v <= start:
            raise ValueError("end_date must be after start_date")
        return v

    @validator("interval")
    def validate_interval(cls, v: str) -> str:
        """Validate that the interval follows supported granularity."""
        allowed = {"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"}
        if v not in allowed:
            raise ValueError(f"interval must be one of {sorted(allowed)}")
        return v


class BacktestResultSchema(BaseModel):
    """
    Schema for the backtest result metrics stored in the database.
    Provides clear documentation and runtime validation.
    """

    total_return: float = Field(
        ...,
        description="Total return expressed as a decimal (e.g., 0.12 for 12%).",
        example=0.12,
    )
    annualized_return: float = Field(
        ...,
        description="Annualized return expressed as a decimal.",
        example=0.15,
    )
    sharpe_ratio: float = Field(
        ...,
        description="Sharpe ratio of the strategy.",
        example=1.2,
    )
    sortino_ratio: float = Field(
        ...,
        description="Sortino ratio of the strategy.",
        example=1.5,
    )
    calmar_ratio: float = Field(
        ...,
        description="Calmar ratio of the strategy.",
        example=0.8,
    )
    max_drawdown: float = Field(
        ...,
        description="Maximum drawdown expressed as a negative decimal.",
        example=-0.2,
    )
    win_rate: float = Field(
        ...,
        description="Proportion of winning trades (0‑1).",
        example=0.55,
    )
    profit_factor: float = Field(
        ...,
        description="Profit factor (gross profit / gross loss).",
        example=1.3,
    )
    total_trades: int = Field(
        ...,
        description="Total number of trades executed.",
        example=42,
    )
    equity_curve: List[float] = Field(
        ...,
        description="Equity curve values over time.",
        example=[100000.0, 101000.0, 102500.0],
    )


async def run_backtest_job(run_id: str | None) -> None:
    """Fetch one queued BacktestRun, execute it, write results back to DB."""
    if not run_id:
        logger.warning("run_backtest_job called with None or empty run_id")
        return

    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun, BacktestResult
    from app.backtest.engine import run_backtest
    from app.backtest.data_loader import fetch_ohlcv
    from app.strategies import STRATEGY_REGISTRY

    async with AsyncSessionLocal() as db:
        run = await db.get(BacktestRun, run_id)
        if not run or run.status != "queued":
            return
        # Validate essential fields
        if not all([run.symbol, run.start_date, run.end_date, run.interval, run.strategy_name]):
            logger.error(f"BacktestRun {run_id} missing required fields")
            run.status = "failed"
            run.error_message = "Missing required backtest parameters"
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()
            return

        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        await db.commit()
        # capture fields before session closes
        symbol = run.symbol
        start_date = run.start_date
        end_date = run.end_date
        interval = run.interval
        strategy_name = run.strategy_name
        initial_equity = (run.params or {}).get("initial_equity", 100_000.0)

    try:
        df = await fetch_ohlcv(symbol=symbol, start=start_date, end=end_date, interval=interval)
        if df.empty:
            raise ValueError(f"No OHLCV data for {symbol} ({start_date}–{end_date})")

        StratClass = STRATEGY_REGISTRY.get(strategy_name)
        if StratClass is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy = StratClass()
        # backtest_signals may be sync or async depending on the strategy
        import inspect
        from app.strategies.base import BacktestSignals as _BSig

        _result = strategy.backtest_signals(df)
        raw_signals = (await _result) if inspect.isawaitable(_result) else _result

        # Convert BacktestSignals → pd.Series[int] expected by run_backtest
        if isinstance(raw_signals, _BSig):
            import numpy as np

            # Ensure entries/exits are not None and have matching index
            entries = raw_signals.entries
            exits = raw_signals.exits
            if entries is None or exits is None:
                raise ValueError("BacktestSignals must contain entries and exits arrays")

            sig = pd.Series(0, index=df.index, dtype=int)
            sig[entries.astype(bool)] = 1
            sig[exits.astype(bool)] = 0
            if raw_signals.short_entries is not None:
                sig[raw_signals.short_entries.astype(bool)] = -1
            signals_series = sig
        else:
            # Expect a pandas Series; guard against empty or mismatched index
            if not isinstance(raw_signals, pd.Series):
                raise TypeError("Strategy signals must be a pandas Series")
            if raw_signals.empty:
                # An empty signal series is treated as no trades
                signals_series = pd.Series(0, index=df.index, dtype=int)
            else:
                # Align index if needed
                signals_series = raw_signals.reindex(df.index, fill_value=0).astype(int)

        metrics = run_backtest(
            signals=signals_series,
            prices=df["close"],
            opens=df["open"],
            volume=df["volume"],
            initial_equity=initial_equity,
        )

        async with AsyncSessionLocal() as db:
            run = await db.get(BacktestRun, run_id)
            if run:
                run.status = "completed"
                run.completed_at = datetime.now(timezone.utc)
                result = BacktestResult(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    total_return=metrics.total_return,
                    annualized_return=metrics.annualized_return,
                    sharpe_ratio=metrics.sharpe,
                    sortino_ratio=metrics.sortino,
                    calmar_ratio=metrics.calmar,
                    max_drawdown=metrics.max_drawdown,
                    win_rate=metrics.win_rate,
                    profit_factor=metrics.profit_factor,
                    total_trades=metrics.num_trades,
                    equity_curve=metrics.equity_curve,
                )
                db.add(result)
                await db.commit()
        logger.info(
            f"Backtest {run_id} complete",
            sharpe=round(metrics.sharpe, 2),
            ret=f"{metrics.total_return:.1%}",
        )

    except Exception as exc:
        logger.error(f"Backtest {run_id} failed: {exc}")
        async with AsyncSessionLocal() as db:
            run = await db.get(BacktestRun, run_id)
            if run:
                run.status = "failed"
                run.error_message = str(exc)[:500]
                run.completed_at = datetime.now(timezone.utc)
                await db.commit()


async def backtest_worker_loop() -> None:
    """Poll for queued BacktestRun rows every 30 s and run them concurrently."""
    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun

    logger.info("Backtest worker started — polling every 30s")
    while True:
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(BacktestRun)
                    .where(BacktestRun.status == "queued")
                    .order_by(BacktestRun.created_at)
                    .limit(5)
                )
                queued = result.scalars().all() or []
                run_ids = [r.id for r in queued if r.id]

            for run_id in run_ids:
                asyncio.create_task(run_backtest_job(run_id))

        except Exception as exc:
            logger.warning(f"Backtest worker poll error: {exc}")

        await asyncio.sleep(30)