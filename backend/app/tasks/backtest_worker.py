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
                queued = result.scalars().all()
                run_ids = [r.id for r in queued]

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

@pytest.mark.asyncio
async def test_run_backtest_job_no_ohlcv(monkeypatch):
    """Empty OHLCV DataFrame should cause the job to fail with a clear error."""
    # Mock BacktestRun instance
    mock_run = MagicMock()
    mock_run.id = "run-1"
    mock_run.status = "queued"
    mock_run.symbol = "FAKE"
    mock_run.start_date = "2020-01-01"
    mock_run.end_date = "2020-01-10"
    mock_run.interval = "1d"
    mock_run.strategy_name = "any"
    mock_run.params = {"initial_equity": 100_000.0}

    # Mock async DB session
    async def mock_get(_, run_id):
        return mock_run if run_id == "run-1" else None

    mock_db = MagicMock()
    mock_db.__aenter__.return_value = mock_db
    mock_db.__aexit__.return_value = AsyncMock()
    mock_db.get = AsyncMock(side_effect=mock_get)
    mock_db.commit = AsyncMock()
    mock_db.add = MagicMock()

    # Patch the imports used inside run_backtest_job
    monkeypatch.setattr("app.database.AsyncSessionLocal", lambda: mock_db)
    monkeypatch.setattr("app.backtest.data_loader.fetch_ohlcv", AsyncMock(return_value=pd.DataFrame()))
    # Ensure any strategy name resolves to a dummy class (won't be used because of empty data)
    monkeypatch.setitem("app.strategies.STRATEGY_REGISTRY", "any", MagicMock())

    await run_backtest_job("run-1")

    assert mock_run.status == "failed"
    assert "No OHLCV data" in mock_run.error_message


@pytest.mark.asyncio
async def test_run_backtest_job_unknown_strategy(monkeypatch):
    """An unknown strategy name should cause the job to fail with an appropriate error."""
    # Minimal non‑empty OHLCV DataFrame
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "close": [1.1, 1.2],
            "volume": [100, 150],
        },
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    mock_run = MagicMock()
    mock_run.id = "run-2"
    mock_run.status = "queued"
    mock_run.symbol = "FAKE"
    mock_run.start_date = "2020-01-01"
    mock_run.end_date = "2020-01-02"
    mock_run.interval = "1d"
    mock_run.strategy_name = "missing_strategy"
    mock_run.params = {}

    async def mock_get(_, run_id):
        return mock_run if run_id == "run-2" else None

    mock_db = MagicMock()
    mock_db.__aenter__.return_value = mock_db
    mock_db.__aexit__.return_value = AsyncMock()
    mock_db.get = AsyncMock(side_effect=mock_get)
    mock_db.commit = AsyncMock()
    mock_db.add = MagicMock()

    monkeypatch.setattr("app.database.AsyncSessionLocal", lambda: mock_db)
    monkeypatch.setattr("app.backtest.data_loader.fetch_ohlcv", AsyncMock(return_value=df))
    # Ensure the registry does NOT contain the requested strategy
    monkeypatch.setattr("app.strategies.STRATEGY_REGISTRY", {}, raising=False)

    await run_backtest_job("run-2")

    assert mock_run.status == "failed"
    assert "Unknown strategy" in mock_run.error_message


@pytest.mark.asyncio
async def test_run_backtest_job_async_signal(monkeypatch):
    """A strategy returning an awaitable signal should be correctly awaited and result in completion."""
    # Simple OHLCV DataFrame with required columns
    df = pd.DataFrame(
        {
            "open": [1.0, 1.2],
            "close": [1.2, 1.3],
            "volume": [200, 250],
        },
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    mock_run = MagicMock()
    mock_run.id = "run-3"
    mock_run.status = "queued"
    mock_run.symbol = "FAKE"
    mock_run.start_date = "2020-01-01"
    mock_run.end_date = "2020-01-02"
    mock_run.interval = "1d"
    mock_run.strategy_name = "async_strat"
    mock_run.params = {}

    async def mock_get(_, run_id):
        return mock_run if run_id == "run-3" else None

    mock_db = MagicMock()
    mock_db.__aenter__.return_value = mock_db
    mock_db.__aexit__.return_value = AsyncMock()
    mock_db.get = AsyncMock(side_effect=mock_get)
    mock_db.commit = AsyncMock()
    mock_db.add = MagicMock()

    # Dummy async strategy
    class AsyncStrategy:
        async def backtest_signals(self, _df):
            # Return a simple pandas Series of signals
            return pd.Series([1, 0], index=_df.index, dtype=int)

    monkeypatch.setattr("app.database.AsyncSessionLocal", lambda: mock_db)
    monkeypatch.setattr("app.backtest.data_loader.fetch_ohlcv", AsyncMock(return_value=df))
    monkeypatch.setitem("app.strategies.STRATEGY_REGISTRY", "async_strat", AsyncStrategy)

    # Mock the backtest engine to return a lightweight metrics object
    class DummyMetrics:
        total_return = 0.05
        annualized_return = 0.07
        sharpe = 1.2
        sortino = 1.0
        calmar = 0.8
        max_drawdown = -0.1
        win_rate = 0.6
        profit_factor = 1.5
        num_trades = 10
        equity_curve = pd.Series([100000, 105000])

    monkeypatch.setattr("app.backtest.engine.run_backtest", lambda **_: DummyMetrics())

    await run_backtest_job("run-3")

    assert mock_run.status == "completed"
    assert mock_run.completed_at is not None
    # Ensure the BacktestResult was added to the session
    assert mock_db.add.called