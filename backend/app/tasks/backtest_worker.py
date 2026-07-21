"""
Backtest worker — polls for queued BacktestRun rows every 30 s and executes them.

Runs as a background asyncio task started from main.py lifespan.
Uses yfinance for free OHLCV data — no broker keys required.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
from sqlalchemy import select

from app.utils.logging import logger


async def _load_backtest_run(run_id: str) -> Any:
    """Load a BacktestRun instance from the DB."""
    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestRun

    async with AsyncSessionLocal() as db:
        return await db.get(BacktestRun, run_id)


def _is_valid_run(run: Any) -> bool:
    """Check that the BacktestRun has all required fields."""
    required = [run.symbol, run.start_date, run.end_date, run.interval, run.strategy_name]
    return all(required)


def _initial_equity(params: Dict[str, Any] | None) -> float:
    """Extract initial equity from run parameters, defaulting to 100 000."""
    return (params or {}).get("initial_equity", 100_000.0)


async def _set_run_status(run: Any, *, status: str, error_message: str | None = None) -> None:
    """Persist a status change for a BacktestRun."""
    from app.database import AsyncSessionLocal

    run.status = status
    if status in {"failed", "completed"}:
        run.completed_at = datetime.now(timezone.utc)
    if error_message:
        run.error_message = error_message[:500]

    async with AsyncSessionLocal() as db:
        db.add(run)
        await db.commit()


async def _fetch_ohlcv(symbol: str, start: Any, end: Any, interval: str) -> pd.DataFrame:
    """Retrieve OHLCV data for the given symbol and period."""
    from app.backtest.data_loader import fetch_ohlcv

    df = await fetch_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
    if df.empty:
        raise ValueError(f"No OHLCV data for {symbol} ({start}–{end})")
    return df


def _resolve_strategy_class(name: str) -> Any:
    """Look up the strategy class in the registry."""
    from app.strategies import STRATEGY_REGISTRY

    strat_cls = STRATEGY_REGISTRY.get(name)
    if strat_cls is None:
        raise ValueError(f"Unknown strategy: {name}")
    return strat_cls


async def _generate_signals(strategy: Any, df: pd.DataFrame) -> pd.Series[int]:
    """Execute the strategy's signal generation and normalise the output."""
    import inspect
    from app.strategies.base import BacktestSignals as _BSig

    raw = strategy.backtest_signals(df)
    raw = await raw if inspect.isawaitable(raw) else raw

    if isinstance(raw, _BSig):
        import numpy as np

        entries = raw.entries
        exits = raw.exits
        if entries is None or exits is None:
            raise ValueError("BacktestSignals must contain entries and exits arrays")

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[entries.astype(bool)] = 1
        sig[exits.astype(bool)] = 0
        if raw.short_entries is not None:
            sig[raw.short_entries.astype(bool)] = -1
        return sig

    if not isinstance(raw, pd.Series):
        raise TypeError("Strategy signals must be a pandas Series")

    if raw.empty:
        return pd.Series(0, index=df.index, dtype=int)

    return raw.reindex(df.index, fill_value=0).astype(int)


def _run_backtest(signals: pd.Series[int], df: pd.DataFrame, initial_equity: float) -> Any:
    """Delegate to the backtest engine."""
    from app.backtest.engine import run_backtest

    return run_backtest(
        signals=signals,
        prices=df["close"],
        opens=df["open"],
        volume=df["volume"],
        initial_equity=initial_equity,
    )


async def _store_backtest_result(run_id: str, metrics: Any) -> None:
    """Persist BacktestResult linked to the given run."""
    from app.database import AsyncSessionLocal
    from app.models.backtest import BacktestResult

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
    async with AsyncSessionLocal() as db:
        db.add(result)
        await db.commit()


async def run_backtest_job(run_id: str | None) -> None:
    """Fetch one queued BacktestRun, execute it, and write results back to DB."""
    if not run_id:
        logger.warning("run_backtest_job called with None or empty run_id")
        return

    # ------------------------------------------------------------------
    # Load and validate the BacktestRun
    # ------------------------------------------------------------------
    run = await _load_backtest_run(run_id)
    if not run or run.status != "queued":
        return

    if not _is_valid_run(run):
        await _set_run_status(
            run,
            status="failed",
            error_message="Missing required backtest parameters",
        )
        return

    # ------------------------------------------------------------------
    # Mark as running and capture needed fields
    # ------------------------------------------------------------------
    run.status = "running"
    run.started_at = datetime.now(timezone.utc)
    async with (await run.__aenter__() if hasattr(run, "__aenter__") else asyncio.sleep(0)):
        # Ensure DB commit before releasing the session
        from app.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            await db.commit()

    symbol = run.symbol
    start_date = run.start_date
    end_date = run.end_date
    interval = run.interval
    strategy_name = run.strategy_name
    equity = _initial_equity(run.params)

    try:
        # ------------------------------------------------------------------
        # Data acquisition
        # ------------------------------------------------------------------
        df = await _fetch_ohlcv(symbol, start_date, end_date, interval)

        # ------------------------------------------------------------------
        # Strategy execution
        # ------------------------------------------------------------------
        StratClass = _resolve_strategy_class(strategy_name)
        strategy = StratClass()
        signals_series = await _generate_signals(strategy, df)

        # ------------------------------------------------------------------
        # Backtest engine
        # ------------------------------------------------------------------
        metrics = _run_backtest(signals_series, df, equity)

        # ------------------------------------------------------------------
        # Persist results
        # ------------------------------------------------------------------
        await _set_run_status(run, status="completed")
        await _store_backtest_result(run_id, metrics)

        logger.info(
            f"Backtest {run_id} complete",
            sharpe=round(metrics.sharpe, 2),
            ret=f"{metrics.total_return:.1%}",
        )
    except Exception as exc:  # pragma: no cover
        logger.error(f"Backtest {run_id} failed: {exc}")
        await _set_run_status(run, status="failed", error_message=str(exc))


async def backtest_worker_loop() -> None:
    """Poll for queued BacktestRun rows every 30 s and run them concurrently."""
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
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Backtest worker poll error: {exc}")

        await asyncio.sleep(30)