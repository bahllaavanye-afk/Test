"""
Backtest worker — polls for queued BacktestRun rows every 30 s and executes them.

Runs as a background asyncio task started from main.py lifespan.
Uses yfinance for free OHLCV data — no broker keys required.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone, date

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, validator, root_validator

from sqlalchemy import select
from app.utils.logging import logger


class BacktestParams(BaseModel):
    """Validated parameters required to execute a backtest."""

    symbol: str = Field(
        ...,
        description="Ticker symbol to backtest.",
        example="AAPL",
    )
    start_date: date = Field(
        ...,
        description="Inclusive start date for the backtest.",
        example="2020-01-01",
    )
    end_date: date = Field(
        ...,
        description="Inclusive end date for the backtest.",
        example="2020-12-31",
    )
    interval: str = Field(
        ...,
        description="Data interval (e.g., '1d', '1h').",
        example="1d",
    )
    strategy_name: str = Field(
        ...,
        description="Registered strategy identifier.",
        example="mean_rev_20_1.5",
    )
    initial_equity: float = Field(
        100_000.0,
        gt=0,
        description="Initial capital for the backtest.",
        example=100000.0,
    )

    @validator("symbol")
    def symbol_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("symbol must not be empty")
        return v

    @validator("interval")
    def interval_allowed(cls, v: str) -> str:
        allowed = {"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"}
        if v not in allowed:
            raise ValueError(f"interval must be one of {allowed}")
        return v

    @root_validator
    def dates_order(cls, values):
        start = values.get("start_date")
        end = values.get("end_date")
        if start and end and start > end:
            raise ValueError("start_date must be before or equal to end_date")
        return values


async def run_backtest_job(run_id: str) -> None:
    """Fetch one queued BacktestRun, execute it, write results back to DB."""
    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun, BacktestResult
    from app.backtest.engine import run_backtest
    from app.backtest.data_loader import fetch_ohlcv
    from app.strategies import STRATEGY_REGISTRY

    async with AsyncSessionLocal() as db:
        run = await db.get(BacktestRun, run_id)
        if not run or run.status != "queued":
            return
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        await db.commit()
        # capture fields before session closes
        raw_symbol = run.symbol
        raw_start_date = run.start_date
        raw_end_date = run.end_date
        raw_interval = run.interval
        raw_strategy_name = run.strategy_name
        raw_initial_equity = (run.params or {}).get("initial_equity", 100_000.0)

    try:
        params = BacktestParams(
            symbol=raw_symbol,
            start_date=raw_start_date,
            end_date=raw_end_date,
            interval=raw_interval,
            strategy_name=raw_strategy_name,
            initial_equity=raw_initial_equity,
        )
    except ValidationError as ve:
        logger.error(f"Backtest {run_id} validation error: {ve}")
        async with AsyncSessionLocal() as db:
            run = await db.get(BacktestRun, run_id)
            if run:
                run.status = "failed"
                run.error_message = str(ve)[:500]
                run.completed_at = datetime.now(timezone.utc)
                await db.commit()
        return

    try:
        df = await fetch_ohlcv(
            symbol=params.symbol,
            start=params.start_date,
            end=params.end_date,
            interval=params.interval,
        )
        if df.empty:
            raise ValueError(
                f"No OHLCV data for {params.symbol} ({params.start_date}–{params.end_date})"
            )

        StratClass = STRATEGY_REGISTRY.get(params.strategy_name)
        if StratClass is None:
            raise ValueError(f"Unknown strategy: {params.strategy_name}")

        strategy = StratClass()
        # backtest_signals may be sync or async depending on the strategy
        import inspect
        from app.strategies.base import BacktestSignals as _BSig

        _result = strategy.backtest_signals(df)
        raw_signals = (await _result) if inspect.isawaitable(_result) else _result

        # Convert BacktestSignals → pd.Series[int] expected by run_backtest
        if isinstance(raw_signals, _BSig):
            import numpy as np

            sig = pd.Series(0, index=df.index, dtype=int)
            sig[raw_signals.entries.astype(bool)] = 1
            sig[raw_signals.exits.astype(bool)] = 0
            if raw_signals.short_entries is not None:
                sig[raw_signals.short_entries.astype(bool)] = -1
            signals_series = sig
        else:
            signals_series = raw_signals  # already a pd.Series

        metrics = run_backtest(
            signals=signals_series,
            prices=df["close"],
            opens=df["open"],
            volume=df["volume"],
            initial_equity=params.initial_equity,
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
                queued = result.scalars().all()
                run_ids = [r.id for r in queued]

            for run_id in run_ids:
                asyncio.create_task(run_backtest_job(run_id))

        except Exception as exc:
            logger.warning(f"Backtest worker poll error: {exc}")

        await asyncio.sleep(30)