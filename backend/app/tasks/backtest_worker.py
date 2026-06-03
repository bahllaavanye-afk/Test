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
