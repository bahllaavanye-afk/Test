"""
Backtest worker — polls for queued BacktestRun rows every 30 s and executes them.

Runs as a background asyncio task started from main.py lifespan.
Uses yfinance for free OHLCV data — no broker keys required.
"""
from __future__ import annotations
import asyncio
import uuid
import pandas as pd
from datetime import datetime, timezone

from sqlalchemy import select
from app.utils.logging import logger


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


# ---------------------------------------------------------------------------
# Unit tests for edge‑case behavior of run_backtest_job
# ---------------------------------------------------------------------------
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Helper mock BacktestRun object
class MockBacktestRun:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", "test-run-id")
        self.status = kwargs.get("status", "queued")
        self.symbol = kwargs.get("symbol")
        self.start_date = kwargs.get("start_date")
        self.end_date = kwargs.get("end_date")
        self.interval = kwargs.get("interval")
        self.strategy_name = kwargs.get("strategy_name")
        self.params = kwargs.get("params", {})
        self.created_at = kwargs.get("created_at", datetime.now(timezone.utc))
        self.error_message = None
        self.completed_at = None
        self.started_at = None


@pytest.mark.asyncio
async def test_run_backtest_job_none_run_id_logs_warning(caplog):
    """Edge case: run_backtest_job receives None as run_id."""
    caplog.set_level("WARNING")
    await run_backtest_job(None)
    assert any("run_backtest_job called with None or empty run_id" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_missing_required_fields_sets_failed():
    """Edge case: required fields missing; should mark run as failed."""
    mock_run = MockBacktestRun(symbol=None, start_date=datetime.now(timezone.utc), end_date=datetime.now(timezone.utc), interval="1d", strategy_name="dummy")
    async_session_mock = AsyncMock()
    async_session_mock.__aenter__.return_value.get.return_value = mock_run
    async_session_mock.__aenter__.return_value.commit = AsyncMock()

    with patch("app.database.AsyncSessionLocal", return_value=async_session_mock):
        await run_backtest_job("any-id")

    assert mock_run.status == "failed"
    assert mock_run.error_message == "Missing required backtest parameters"


@pytest.mark.asyncio
async def test_empty_ohlcv_triggers_failure():
    """Edge case: fetch_ohlcv returns empty DataFrame, causing failure."""
    mock_run = MockBacktestRun(
        symbol="FAKE",
        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2020, 1, 10, tzinfo=timezone.utc),
        interval="1d",
        strategy_name="dummy",
    )
    async_session_mock = AsyncMock()
    async_session_mock.__aenter__.return_value.get.return_value = mock_run
    async_session_mock.__aenter__.return_value.commit = AsyncMock()

    # Mock fetch_ohlcv to return empty DataFrame
    async_fetch = AsyncMock(return_value=pd.DataFrame())
    # Mock strategy registry to return a dummy strategy that would never be called
    dummy_strategy = MagicMock()
    dummy_strategy.backtest_signals.return_value = pd.Series([], dtype=int)

    with patch("app.database.AsyncSessionLocal", return_value=async_session_mock), \
         patch("app.backtest.data_loader.fetch_ohlcv", async_fetch), \
         patch.dict("app.strategies.STRATEGY_REGISTRY", {"dummy": lambda: dummy_strategy}):
        await run_backtest_job("run-id-empty-data")

    assert mock_run.status == "failed"
    assert "No OHLCV data" in mock_run.error_message