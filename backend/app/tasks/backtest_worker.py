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


async def _fetch_queued_run(run_id: str):
    """Retrieve a queued BacktestRun and mark it as running."""
    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun

    async with AsyncSessionLocal() as db:
        run = await db.get(BacktestRun, run_id)
        if not run or run.status != "queued":
            return None
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        await db.commit()
        # capture needed fields before the session closes
        return {
            "run": run,
            "symbol": run.symbol,
            "start_date": run.start_date,
            "end_date": run.end_date,
            "interval": run.interval,
            "strategy_name": run.strategy_name,
            "initial_equity": (run.params or {}).get("initial_equity", 100_000.0),
        }


async def _load_ohlcv(symbol: str, start, end, interval) -> pd.DataFrame:
    """Fetch OHLCV data and validate its presence."""
    from app.backtest.data_loader import fetch_ohlcv

    df = await fetch_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
    if df.empty:
        raise ValueError(f"No OHLCV data for {symbol} ({start}–{end})")
    return df


def _resolve_strategy(strategy_name: str):
    """Retrieve strategy class from registry."""
    from app.strategies import STRATEGY_REGISTRY

    StratClass = STRATEGY_REGISTRY.get(strategy_name)
    if StratClass is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return StratClass()


async def _execute_strategy(strategy, df: pd.DataFrame) -> pd.Series:
    """Run strategy backtest_signals and return a pandas Series of signals."""
    import inspect
    from app.strategies.base import BacktestSignals as _BSig

    raw = strategy.backtest_signals(df)
    raw_signals = await raw if inspect.isawaitable(raw) else raw

    if isinstance(raw_signals, _BSig):
        import numpy as np

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[raw_signals.entries.astype(bool)] = 1
        sig[raw_signals.exits.astype(bool)] = 0
        if raw_signals.short_entries is not None:
            sig[raw_signals.short_entries.astype(bool)] = -1
        return sig
    return raw_signals  # already a pd.Series


async def _store_result(run_id: str, metrics) -> None:
    """Persist BacktestResult and update run status to completed."""
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
    """Mark the run as failed and record the error message."""
    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun

    async with AsyncSessionLocal() as db:
        run = await db.get(BacktestRun, run_id)
        if run:
            run.status = "failed"
            run.error_message = str(exc)[:500]
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()


async def run_backtest_job(run_id: str) -> None:
    """Fetch one queued BacktestRun, execute it, write results back to DB."""
    from app.backtest.engine import run_backtest

    run_info = await _fetch_queued_run(run_id)
    if not run_info:
        return

    try:
        df = await _load_ohlcv(
            symbol=run_info["symbol"],
            start=run_info["start_date"],
            end=run_info["end_date"],
            interval=run_info["interval"],
        )

        strategy = _resolve_strategy(run_info["strategy_name"])
        signals_series = await _execute_strategy(strategy, df)

        metrics = run_backtest(
            signals=signals_series,
            prices=df["close"],
            opens=df["open"],
            volume=df["volume"],
            initial_equity=run_info["initial_equity"],
        )

        await _store_result(run_id, metrics)

        logger.info(
            f"Backtest {run_id} complete",
            sharpe=round(metrics.sharpe, 2),
            ret=f"{metrics.total_return:.1%}",
        )
    except Exception as exc:
        logger.error(f"Backtest {run_id} failed: {exc}")
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
                queued = result.scalars().all()
                run_ids = [r.id for r in queued]

            for run_id in run_ids:
                asyncio.create_task(run_backtest_job(run_id))
        except Exception as exc:
            logger.warning(f"Backtest worker poll error: {exc}")

        await asyncio.sleep(30)