"""Backtest trigger and result retrieval endpoints."""
from datetime import date, datetime, timezone
import uuid
from typing import Optional, List

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user
from app.api.limiter import limiter
from app.backtest.stress_test import STRESS_SCENARIOS
from app.database import get_db
from app.models.backtest import BacktestRun, BacktestResult
from app.models.user import User
from app.strategies import STRATEGY_REGISTRY

router = APIRouter(prefix="/backtests", tags=["backtests"])


class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    interval: str = "1d"
    start_date: date
    end_date: date
    initial_equity: float = 100_000


class WalkForwardRequest(BaseModel):
    strategy_name: str
    symbol: str
    interval: str = "1d"
    start_date: date
    end_date: date
    train_years: int = 2
    test_months: int = 6
    initial_equity: float = 100_000


class BacktestOut(BaseModel):
    id: str
    strategy_name: str
    symbol: str
    interval: str
    status: str
    sharpe: float | None = None
    sortino: float | None = None
    calmar: float | None = None
    max_drawdown: float | None = None
    total_return: float | None = None
    annualized_return: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    total_trades: int | None = None
    equity_curve: list | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_run(cls, run: BacktestRun) -> "BacktestOut":
        result = run.result
        return cls(
            id=run.id,
            strategy_name=run.strategy_name,
            symbol=run.symbol,
            interval=run.interval,
            status=run.status,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
            error_message=run.error_message,
            sharpe=float(result.sharpe_ratio) if result and result.sharpe_ratio is not None else None,
            sortino=float(result.sortino_ratio) if result and result.sortino_ratio is not None else None,
            calmar=float(result.calmar_ratio) if result and result.calmar_ratio is not None else None,
            max_drawdown=float(result.max_drawdown) if result and result.max_drawdown is not None else None,
            total_return=float(result.total_return) if result and result.total_return is not None else None,
            annualized_return=float(result.annualized_return) if result and result.annualized_return is not None else None,
            win_rate=float(result.win_rate) if result and result.win_rate is not None else None,
            profit_factor=float(result.profit_factor) if result and result.profit_factor is not None else None,
            total_trades=result.total_trades if result else None,
            equity_curve=result.equity_curve if result else None,
        )


async def _run_backtest_task(
    run_id: str,
    strategy_name: str,
    symbol: str,
    interval: str,
    start_date: Optional[date],
    end_date: Optional[date],
    initial_equity: Optional[float],
) -> None:
    """Background task: fetch OHLCV, run strategy.backtest_signals(), pass to engine."""
    from app.backtest.data_loader import fetch_ohlcv
    from app.backtest.engine import run_backtest
    from app.database import AsyncSessionLocal

    # Defensive checks for None / invalid inputs
    if not all([run_id, strategy_name, symbol, interval, start_date, end_date, initial_equity]):
        return

    async with AsyncSessionLocal() as db:
        # Mark as running
        run_q = await db.execute(select(BacktestRun).where(BacktestRun.id == run_id))
        run = run_q.scalar_one_or_none()
        if run is None:
            return
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        await db.commit()

        try:
            # Resolve strategy class
            strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_cls is None:
                run.status = "failed"
                run.error_message = f"Unknown strategy: {strategy_name}"
                run.completed_at = datetime.now(timezone.utc)
                await db.commit()
                return

            market_type = getattr(strategy_cls, "market_type", "equity")

            # Load OHLCV data via yfinance (free, no API key)
            df = await fetch_ohlcv(symbol, start_date, end_date, interval, market_type)
            if df is None or df.empty or len(df) < 20:
                run.status = "failed"
                run.error_message = f"Insufficient data for {symbol} ({interval})"
                run.completed_at = datetime.now(timezone.utc)
                await db.commit()
                return

            # Generate backtest signals via the strategy's backtest_signals()
            strategy = strategy_cls()
            bt_signals = strategy.backtest_signals(df)

            # Convert BacktestSignals → numeric signal series (-1/0/+1)
            signals = pd.Series(0.0, index=df.index)

            if getattr(bt_signals, "entries", None):
                signals[bt_signals.entries] = 1.0
            if getattr(bt_signals, "exits", None):
                signals[bt_signals.exits] = 0.0
            if getattr(bt_signals, "short_entries", None):
                signals[bt_signals.short_entries] = -1.0
            if getattr(bt_signals, "short_exits", None):
                short_exits = bt_signals.short_exits
                # Ensure we only reset positions that are currently short
                signals[short_exits & (signals == -1.0)] = 0.0

            # Run vectorized backtest engine
            opens = df["open"] if "open" in df.columns else None
            volume = df["volume"] if "volume" in df.columns else None
            metrics = run_backtest(
                signals=signals,
                prices=df["close"],
                opens=opens,
                volume=volume,
                initial_equity=initial_equity,
            )

            # Persist result to DB
            result = BacktestResult(
                id=str(uuid.uuid4()),
                run_id=run_id,
                total_return=round(metrics.total_return, 6),
                annualized_return=round(metrics.annualized_return, 6),
                sharpe_ratio=round(metrics.sharpe, 4),
                sortino_ratio=round(metrics.sortino, 4),
                calmar_ratio=round(metrics.calmar, 4),
                max_drawdown=round(metrics.max_drawdown, 4),
                win_rate=round(metrics.win_rate, 4),
                profit_factor=round(metrics.profit_factor, 4),
                total_trades=metrics.num_trades,
                equity_curve=metrics.equity_curve[:500],  # cap payload size
            )
            db.add(result)

            run.status = "done"
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()

        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)[:500]
            run.completed_at = datetime.now(timezone.utc)
            try:
                await db.commit()
            except Exception:
                pass


async def _run_walk_forward_task(
    run_id: str,
    strategy_name: str,
    symbol: str,
    interval: str,
    start_date: Optional[date],
    end_date: Optional[date],
    train_years: int,
    test_months: int,
    initial_equity: Optional[float],
) -> None:
    """Background task: walk‑forward validation using strategy.backtest_signals()."""
    from app.backtest.data_loader import fetch_ohlcv
    from app.backtest.walk_forward import walk_forward
    from app.database import AsyncSessionLocal

    # Defensive checks for None / invalid inputs
    if not all([run_id, strategy_name, symbol, interval, start_date, end_date, initial_equity]):
        return

    async with AsyncSessionLocal() as db:
        run_q = await db.execute(select(BacktestRun).where(BacktestRun.id == run_id))
        run = run_q.scalar_one_or_none()
        if run is None:
            return

        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        await db.commit()

        try:
            strategy_cls = STRATEGY_REGISTRY.get(strategy_name)
            if strategy_cls is None:
                run.status = "failed"
                run.error_message = f"Unknown strategy: {strategy_name}"
                run.completed_at = datetime.now(timezone.utc)
                await db.commit()
                return

            market_type = getattr(strategy_cls, "market_type", "equity")
            df = await fetch_ohlcv(symbol, start_date, end_date, interval, market_type)
            if df is None or df.empty or len(df) < 30:
                run.status = "failed"
                run.error_message = f"Insufficient data for walk‑forward on {symbol}"
                run.completed_at = datetime.now(timezone.utc)
                await db.commit()
                return

            strategy = strategy_cls()
            metrics = walk_forward(
                strategy=strategy,
                data=df,
                train_years=train_years,
                test_months=test_months,
                initial_equity=initial_equity,
            )

            result = BacktestResult(
                id=str(uuid.uuid4()),
                run_id=run_id,
                total_return=round(metrics.total_return, 6),
                annualized_return=round(metrics.annualized_return, 6),
                sharpe_ratio=round(metrics.sharpe, 4),
                sortino_ratio=round(metrics.sortino, 4),
                calmar_ratio=round(metrics.calmar, 4),
                max_drawdown=round(metrics.max_drawdown, 4),
                win_rate=round(metrics.win_rate, 4),
                profit_factor=round(metrics.profit_factor, 4),
                total_trades=metrics.num_trades,
                equity_curve=metrics.equity_curve[:500],
            )
            db.add(result)

            run.status = "done"
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()
        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)[:500]
            run.completed_at = datetime.now(timezone.utc)
            try:
                await db.commit()
            except Exception:
                pass

@router.post("/", response_model=BacktestOut, dependencies=[Depends(limiter)])
async def trigger_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Validate request dates
    if request.start_date >= request.end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")
    if request.initial_equity <= 0:
        raise HTTPException(status_code=400, detail="initial_equity must be positive")

    # Create a new BacktestRun entry
    run = BacktestRun(
        id=str(uuid.uuid4()),
        user_id=user.id,
        strategy_name=request.strategy_name,
        symbol=request.symbol,
        interval=request.interval,
        start_date=request.start_date,
        end_date=request.end_date,
        initial_equity=request.initial_equity,
        status="queued",
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(
        _run_backtest_task,
        run.id,
        request.strategy_name,
        request.symbol,
        request.interval,
        request.start_date,
        request.end_date,
        request.initial_equity,
    )
    return BacktestOut.from_run(run)


@router.post("/walk-forward", response_model=BacktestOut, dependencies=[Depends(limiter)])
async def trigger_walk_forward(
    request: WalkForwardRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Basic validation
    if request.start_date >= request.end_date:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")
    if request.initial_equity <= 0:
        raise HTTPException(status_code=400, detail="initial_equity must be positive")
    if request.train_years <= 0 or request.test_months <= 0:
        raise HTTPException(status_code=400, detail="train_years and test_months must be positive")

    run = BacktestRun(
        id=str(uuid.uuid4()),
        user_id=user.id,
        strategy_name=request.strategy_name,
        symbol=request.symbol,
        interval=request.interval,
        start_date=request.start_date,
        end_date=request.end_date,
        initial_equity=request.initial_equity,
        status="queued",
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(
        _run_walk_forward_task,
        run.id,
        request.strategy_name,
        request.symbol,
        request.interval,
        request.start_date,
        request.end_date,
        request.train_years,
        request.test_months,
        request.initial_equity,
    )
    return BacktestOut.from_run(run)


@router.get("/{run_id}", response_model=BacktestOut, dependencies=[Depends(limiter)])
async def get_backtest(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(BacktestRun)
        .options(selectinload(BacktestRun.result))
        .where(BacktestRun.id == run_id, BacktestRun.user_id == user.id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return BacktestOut.from_run(run)