"""
Backtest worker — polls for queued BacktestRun rows every POLL_INTERVAL_SECONDS s and executes them.

Runs as a background asyncio task started from main.py lifespan.
Uses yfinance for free OHLCV data — no broker keys required.
"""
from __future__ import annotations

import asyncio
import uuid
import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select

from app.utils.logging import logger

# Configuration constants
POLL_INTERVAL_SECONDS: int = 30
FETCH_LIMIT: int = 5
DEFAULT_INITIAL_EQUITY: float = 100_000.0
MAX_ERROR_MESSAGE_LENGTH: int = 500

# Status strings
STATUS_QUEUED: str = "queued"
STATUS_RUNNING: str = "running"
STATUS_COMPLETED: str = "completed"
STATUS_FAILED: str = "failed"

# Log / error message templates
MSG_RUN_BACKTEST_JOB_NONE_ID = "run_backtest_job called with None or empty run_id"
MSG_MISSING_PARAMS = "Missing required backtest parameters"
MSG_NO_OHLCV_DATA = "No OHLCV data for {symbol} ({start_date}–{end_date})"
MSG_UNKNOWN_STRATEGY = "Unknown strategy: {strategy_name}"
MSG_SIGNALS_MUST_CONTAIN = "BacktestSignals must contain entries and exits arrays"
MSG_STRATEGY_SIGNALS_TYPE = "Strategy signals must be a pandas Series"
MSG_WORKER_STARTED = "Backtest worker started — polling every {interval}s"
MSG_WORKER_POLL_ERROR = "Backtest worker poll error: {exc}"
MSG_BACKTEST_COMPLETE = "Backtest {run_id} complete"
MSG_BACKTEST_FAILED = "Backtest {run_id} failed: {exc}"


async def run_backtest_job(run_id: str | None) -> None:
    """Fetch one queued BacktestRun, execute it, write results back to DB."""
    if not run_id:
        logger.warning(MSG_RUN_BACKTEST_JOB_NONE_ID)
        return

    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun, BacktestResult
    from app.backtest.engine import run_backtest
    from app.backtest.data_loader import fetch_ohlcv
    from app.strategies import STRATEGY_REGISTRY

    async with AsyncSessionLocal() as db:
        run = await db.get(BacktestRun, run_id)
        if not run or run.status != STATUS_QUEUED:
            return
        # Validate essential fields
        if not all([run.symbol, run.start_date, run.end_date, run.interval, run.strategy_name]):
            logger.error(f"BacktestRun {run_id} {MSG_MISSING_PARAMS}")
            run.status = STATUS_FAILED
            run.error_message = MSG_MISSING_PARAMS
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()
            return

        run.status = STATUS_RUNNING
        run.started_at = datetime.now(timezone.utc)
        await db.commit()
        # capture fields before session closes
        symbol = run.symbol
        start_date = run.start_date
        end_date = run.end_date
        interval = run.interval
        strategy_name = run.strategy_name
        initial_equity = (run.params or {}).get("initial_equity", DEFAULT_INITIAL_EQUITY)

    try:
        df = await fetch_ohlcv(symbol=symbol, start=start_date, end=end_date, interval=interval)
        if df.empty:
            raise ValueError(MSG_NO_OHLCV_DATA.format(symbol=symbol, start_date=start_date, end_date=end_date))

        StratClass = STRATEGY_REGISTRY.get(strategy_name)
        if StratClass is None:
            raise ValueError(MSG_UNKNOWN_STRATEGY.format(strategy_name=strategy_name))

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
                raise ValueError(MSG_SIGNALS_MUST_CONTAIN)

            sig = pd.Series(0, index=df.index, dtype=int)
            sig[entries.astype(bool)] = 1
            sig[exits.astype(bool)] = 0
            if raw_signals.short_entries is not None:
                sig[raw_signals.short_entries.astype(bool)] = -1
            signals_series = sig
        else:
            # Expect a pandas Series; guard against empty or mismatched index
            if not isinstance(raw_signals, pd.Series):
                raise TypeError(MSG_STRATEGY_SIGNALS_TYPE)
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
                run.status = STATUS_COMPLETED
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
            MSG_BACKTEST_COMPLETE.format(run_id=run_id),
            sharpe=round(metrics.sharpe, 2),
            ret=f"{metrics.total_return:.1%}",
        )

    except Exception as exc:
        logger.error(MSG_BACKTEST_FAILED.format(run_id=run_id, exc=exc))
        async with AsyncSessionLocal() as db:
            run = await db.get(BacktestRun, run_id)
            if run:
                run.status = STATUS_FAILED
                run.error_message = str(exc)[:MAX_ERROR_MESSAGE_LENGTH]
                run.completed_at = datetime.now(timezone.utc)
                await db.commit()


async def backtest_worker_loop() -> None:
    """Poll for queued BacktestRun rows every POLL_INTERVAL_SECONDS s and run them concurrently."""
    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun

    logger.info(MSG_WORKER_STARTED.format(interval=POLL_INTERVAL_SECONDS))
    while True:
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(BacktestRun)
                    .where(BacktestRun.status == STATUS_QUEUED)
                    .order_by(BacktestRun.created_at)
                    .limit(FETCH_LIMIT)
                )
                queued = result.scalars().all() or []
                run_ids = [r.id for r in queued if r.id]

            for run_id in run_ids:
                asyncio.create_task(run_backtest_job(run_id))

        except Exception as exc:
            logger.warning(MSG_WORKER_POLL_ERROR.format(exc=exc))

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Unit tests for edge‑case handling in run_backtest_job
# ---------------------------------------------------------------------------
import pytest

@pytest.mark.asyncio
async def test_run_backtest_job_none_id(monkeypatch, caplog):
    """Ensure the function returns early and logs a warning when run_id is None."""
    caplog.set_level(logging.WARNING, logger.name)
    await run_backtest_job(None)
    assert any(MSG_RUN_BACKTEST_JOB_NONE_ID in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_missing_params_triggers_failure(monkeypatch):
    """A BacktestRun missing required fields should be marked FAILED with proper error."""
    # Minimal mock BacktestRun with missing start_date
    class MockRun:
        def __init__(self):
            self.id = "test-id"
            self.status = STATUS_QUEUED
            self.symbol = "AAPL"
            self.start_date = None  # missing
            self.end_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
            self.interval = "1d"
            self.strategy_name = "dummy"
            self.params = {}
            self.error_message = None
            self.completed_at = None

    class MockSession:
        def __init__(self, run):
            self.run = run

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, model, run_id):
            return self.run if run_id == self.run.id else None

        async def commit(self):
            pass

    mock_run = MockRun()
    monkeypatch.setattr("app.database.AsyncSessionLocal", lambda: MockSession(mock_run))
    await run_backtest_job(mock_run.id)

    assert mock_run.status == STATUS_FAILED
    assert mock_run.error_message == MSG_MISSING_PARAMS


@pytest.mark.asyncio
async def test_empty_ohlcv_data_results_in_failure(monkeypatch):
    """When fetch_ohlcv returns an empty DataFrame, the run should fail with a specific message."""
    # Mock run with all required fields present
    class MockRun:
        def __init__(self):
            self.id = "empty-data-id"
            self.status = STATUS_QUEUED
            self.symbol = "MSFT"
            self.start_date = datetime(2022, 1, 1, tzinfo=timezone.utc)
            self.end_date = datetime(2022, 12, 31, tzinfo=timezone.utc)
            self.interval = "1d"
            self.strategy_name = "dummy"
            self.params = {}
            self.error_message = None
            self.completed_at = None

    class MockSession:
        def __init__(self, run):
            self.run = run

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, model, run_id):
            return self.run if run_id == self.run.id else None

        async def commit(self):
            pass

        async def add(self, obj):
            pass

    mock_run = MockRun()
    monkeypatch.setattr("app.database.AsyncSessionLocal", lambda: MockSession(mock_run))

    async def mock_fetch_ohlcv(symbol, start, end, interval):
        return pd.DataFrame()  # empty DataFrame

    monkeypatch.setattr("app.backtest.data_loader.fetch_ohlcv", mock_fetch_ohlcv)

    await run_backtest_job(mock_run.id)

    assert mock_run.status == STATUS_FAILED
    expected_msg = MSG_NO_OHLCV_DATA.format(symbol=mock_run.symbol,
                                            start_date=mock_run.start_date,
                                            end_date=mock_run.end_date)
    assert expected_msg in mock_run.error_message