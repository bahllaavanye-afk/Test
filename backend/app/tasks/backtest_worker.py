"""
Backtest worker — polls for queued BacktestRun rows every 30 s and executes them.

Runs as a background asyncio task started from main.py lifespan.
Uses yfinance for free OHLCV data — no broker keys required.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sqlalchemy import select
from app.utils.logging import logger


def _apply_confirmation_filters(signals: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Tighten entry conditions and improve exit logic.

    - Require an entry signal to be confirmed by at least one additional bar
      with price moving in the same direction.
    - Filter out entries on unusually low volume (below the median).
    - Ensure exit signals only fire when a position is actually open.
    - Preserve the original signal type (1 for long, -1 for short, 0 for flat).

    Parameters
    ----------
    signals: pd.Series[int]
        Raw integer signals indexed like ``df`` (1 = long entry, -1 = short entry,
        0 = flat/exit).
    df: pd.DataFrame
        OHLCV data used for confirmation checks. Must contain ``open``, ``close``,
        and ``volume`` columns.

    Returns
    -------
    pd.Series[int]
        Filter‑adjusted signals.
    """
    # Copy to avoid mutating caller data
    filtered = signals.copy()

    # 1️⃣ Volume filter – drop entries where volume is below the median
    median_vol = df["volume"].median()
    low_vol_mask = df["volume"] < median_vol
    filtered[(filtered != 0) & low_vol_mask] = 0

    # 2️⃣ Confirmation filter – require price movement in the same direction
    #    on the following bar for a true entry.
    #    For longs we expect next close > next open; for shorts the opposite.
    next_close = df["close"].shift(-1)
    next_open = df["open"].shift(-1)

    long_entries = (filtered == 1) & (next_close <= next_open)
    short_entries = (filtered == -1) & (next_close >= next_open)

    filtered[long_entries | short_entries] = 0

    # 3️⃣ Exit sanity – an exit (0) should only be emitted when a position is open.
    #    We compute a running position and zero out spurious exits.
    position = 0
    for idx, sig in filtered.items():
        if sig == 1:
            position = 1
        elif sig == -1:
            position = -1
        elif sig == 0 and position == 0:
            # No open position, suppress exit
            filtered.at[idx] = np.nan
        elif sig == 0:
            position = 0

    # Replace suppressed exits with no‑action (0) to keep dtype int
    filtered = filtered.fillna(0).astype(int)

    return filtered


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

        # Capture fields before session closes
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
                signals_series = pd.Series(0, index=df.index, dtype=int)
            else:
                signals_series = raw_signals.reindex(df.index, fill_value=0).astype(int)

        # -----------------------------------------------------------------
        # Apply tightened entry/exit logic
        # -----------------------------------------------------------------
        signals_series = _apply_confirmation_filters(signals_series, df)

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

        except Exception as exc:
            logger.warning(f"Backtest worker poll error: {exc}")

        await asyncio.sleep(30)