"""
Backtest worker — polls for queued BacktestRun rows every 30 s and executes them.

Runs as a background asyncio task started from main.py lifespan.
Uses yfinance for free OHLCV data — no broker keys required.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select

from app.utils.logging import logger


async def _validate_and_prepare_run(run_id: str) -> dict | None:
    """Fetch BacktestRun, validate required fields, and update status to running.

    Returns a dict with run data needed for execution or None if validation fails.
    """
    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun

    async with AsyncSessionLocal() as db:
        run = await db.get(BacktestRun, run_id)
        if not run or run.status != "queued":
            return None

        required = [run.symbol, run.start_date, run.end_date, run.interval, run.strategy_name]
        if not all(required):
            logger.error(f"BacktestRun {run_id} missing required fields")
            run.status = "failed"
            run.error_message = "Missing required backtest parameters"
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()
            return None

        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        await db.commit()

        return {
            "symbol": run.symbol,
            "start_date": run.start_date,
            "end_date": run.end_date,
            "interval": run.interval,
            "strategy_name": run.strategy_name,
            "initial_equity": (run.params or {}).get("initial_equity", 100_000.0),
        }


async def _fetch_ohlcv_data(symbol: str, start, end, interval: str) -> pd.DataFrame:
    """Retrieve OHLCV data for the given parameters.

    Raises:
        ValueError: If the returned DataFrame is empty.
    """
    from app.backtest.data_loader import fetch_ohlcv

    df = await fetch_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
    if df.empty:
        raise ValueError(f"No OHLCV data for {symbol} ({start}–{end})")
    return df


def _convert_backtest_signals(raw_signals, df: pd.DataFrame) -> pd.Series:
    """Normalize strategy output to a pandas Series of int signals."""
    from app.strategies.base import BacktestSignals as _BSig

    if isinstance(raw_signals, _BSig):
        import numpy as np

        entries = raw_signals.entries
        exits = raw_signals.exits
        if entries is None or exits is None:
            raise ValueError("BacktestSignals must contain entries and exits arrays")

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[entries.astype(bool)] = 1
        sig[exits.astype(bool)] = 0
        if raw_signals.short_entries is not None:
            sig[raw_signals.short_entries.astype(bool)] = -1
        return sig

    if not isinstance(raw_signals, pd.Series):
        raise TypeError("Strategy signals must be a pandas Series")

    if raw_signals.empty:
        return pd.Series(0, index=df.index, dtype=int)

    return raw_signals.reindex(df.index, fill_value=0).astype(int)


async def _run_strategy_and_backtest(
    strategy_name: str,
    df: pd.DataFrame,
    initial_equity: float,
) -> tuple[pd.Series, object]:
    """Execute the strategy's signal generation and run the backtest engine.

    Returns the signals series and the metrics object.
    """
    from app.backtest.engine import run_backtest
    from app.strategies import STRATEGY_REGISTRY

    StratClass = STRATEGY_REGISTRY.get(strategy_name)
    if StratClass is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy = StratClass()
    result = strategy.backtest_signals(df)

    import inspect

    raw_signals = await result if inspect.isawaitable(result) else result
    signals_series = _convert_backtest_signals(raw_signals, df)

    metrics = run_backtest(
        signals=signals_series,
        prices=df["close"],
        opens=df["open"],
        volume=df["volume"],
        initial_equity=initial_equity,
    )
    return signals_series, metrics


async def _store_backtest_result(run_id: str, metrics) -> None:
    """Persist BacktestResult and update BacktestRun status to completed."""
    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun, BacktestResult

    async with AsyncSessionLocal() as db:
        run = await db.get(BacktestRun, run_id)
        if not run:
            return

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


async def _handle_failure(run_id: str, exc: Exception) -> None:
    """Log the error and mark the BacktestRun as failed."""
    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun

    logger.error(f"Backtest {run_id} failed: {exc}")
    async with AsyncSessionLocal() as db:
        run = await db.get(BacktestRun, run_id)
        if run:
            run.status = "failed"
            run.error_message = str(exc)[:500]
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()


async def run_backtest_job(run_id: str | None) -> None:
    """Fetch one queued BacktestRun, execute it, and write results back to DB."""
    if not run_id:
        logger.warning("run_backtest_job called with None or empty run_id")
        return

    run_data = await _validate_and_prepare_run(run_id)
    if not run_data:
        return

    try:
        df = await _fetch_ohlcv_data(
            symbol=run_data["symbol"],
            start=run_data["start_date"],
            end=run_data["end_date"],
            interval=run_data["interval"],
        )

        _, metrics = await _run_strategy_and_backtest(
            strategy_name=run_data["strategy_name"],
            df=df,
            initial_equity=run_data["initial_equity"],
        )

        await _store_backtest_result(run_id, metrics)

        logger.info(
            f"Backtest {run_id} complete",
            sharpe=round(metrics.sharpe, 2),
            ret=f"{metrics.total_return:.1%}",
        )
    except Exception as exc:  # pragma: no cover
        await _handle_failure(run_id, exc)


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