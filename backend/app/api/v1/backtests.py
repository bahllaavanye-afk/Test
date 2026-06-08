"""Backtest trigger and result retrieval endpoints."""
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.database import get_db
from app.api.deps import get_current_user
from app.api.limiter import limiter
from app.models.backtest import BacktestRun, BacktestResult
from app.models.user import User
from app.backtest.stress_test import STRESS_SCENARIOS
from pydantic import BaseModel, ConfigDict
from datetime import date, datetime, timezone
import uuid

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
    def from_run(cls, run) -> "BacktestOut":
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


async def _run_backtest_task(run_id: str, strategy_name: str, symbol: str,
                              interval: str, start_date: date, end_date: date,
                              initial_equity: float) -> None:
    """Background task: fetch OHLCV, run strategy.backtest_signals(), pass to engine."""
    from app.database import AsyncSessionLocal
    from app.backtest.data_loader import fetch_ohlcv
    from app.backtest.engine import run_backtest
    from app.strategies import STRATEGY_REGISTRY
    import pandas as pd

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
            signals[bt_signals.entries] = 1.0
            signals[bt_signals.exits] = 0.0
            if bt_signals.short_entries is not None:
                signals[bt_signals.short_entries] = -1.0
            if bt_signals.short_exits is not None:
                signals[bt_signals.short_exits & (signals == -1.0)] = 0.0

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


async def _run_walk_forward_task(run_id: str, strategy_name: str, symbol: str,
                                  interval: str, start_date: date, end_date: date,
                                  train_years: int, test_months: int,
                                  initial_equity: float) -> None:
    """Background task: walk-forward validation using strategy.backtest_signals()."""
    from app.database import AsyncSessionLocal
    from app.backtest.data_loader import fetch_ohlcv
    from app.backtest.walk_forward import walk_forward
    from app.strategies import STRATEGY_REGISTRY
    import pandas as pd
    import statistics

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
            if df is None or df.empty or len(df) < (train_years * 252 + test_months * 21):
                run.status = "failed"
                run.error_message = "Insufficient data for walk-forward validation"
                run.completed_at = datetime.now(timezone.utc)
                await db.commit()
                return

            strategy = strategy_cls()

            def signals_fn(train_prices, test_prices):
                """Generate signals for the test window using the strategy."""
                test_df = df.loc[test_prices.index]
                bt = strategy.backtest_signals(test_df)
                sig = pd.Series(0.0, index=test_df.index)
                sig[bt.entries] = 1.0
                sig[bt.exits] = 0.0
                if bt.short_entries is not None:
                    sig[bt.short_entries] = -1.0
                return sig

            wf_result = walk_forward(
                signals_fn=signals_fn,
                prices=df["close"],
                train_years=train_years,
                test_months=test_months,
                initial_equity=initial_equity,
            )

            sharpes = [
                w["sharpe"] for w in wf_result.windows
                if "sharpe" in w and w["sharpe"] is not None
            ]
            drawdowns = [
                w["max_drawdown"] for w in wf_result.windows
                if "max_drawdown" in w and w["max_drawdown"] is not None
            ]

            result = BacktestResult(
                id=str(uuid.uuid4()),
                run_id=run_id,
                total_return=round(
                    wf_result.combined_equity[-1]["equity"] / initial_equity - 1, 6
                ) if wf_result.combined_equity else 0.0,
                annualized_return=None,
                sharpe_ratio=round(statistics.mean(sharpes), 4) if sharpes else None,
                sortino_ratio=None,
                calmar_ratio=None,
                max_drawdown=round(min(drawdowns), 4) if drawdowns else None,
                win_rate=None,
                profit_factor=None,
                total_trades=sum(w.get("num_trades", 0) for w in wf_result.windows),
                equity_curve=wf_result.combined_equity[:500],
                trades_log=wf_result.windows,
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


@router.get("/scenarios")
async def list_stress_scenarios(
    current_user: User = Depends(get_current_user),
):
    """Return all built-in historical stress-test scenarios."""
    return [
        {
            "id": s.name,
            "label": s.label,
            "start": s.start.isoformat(),
            "end": s.end.isoformat(),
            "description": s.description,
        }
        for s in STRESS_SCENARIOS
    ]


@router.get("/")
async def list_backtests(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(BacktestRun).where(BacktestRun.user_id == current_user.id)
        .options(selectinload(BacktestRun.result))
        .order_by(BacktestRun.created_at.desc()).limit(50)
    )
    runs = result.scalars().all()
    return [BacktestOut.from_run(r) for r in runs]


@router.get("/{run_id}")
async def get_backtest(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Poll status of a specific backtest run."""
    q = await db.execute(
        select(BacktestRun)
        .where(BacktestRun.id == run_id, BacktestRun.user_id == current_user.id)
        .options(selectinload(BacktestRun.result))
    )
    run = q.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Backtest run not found")
    return BacktestOut.from_run(run)


@router.post("/run")
@limiter.limit("5/minute")
async def trigger_backtest_run(
    request: Request,
    body: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    POST /backtests/run — trigger a full vectorized backtest.

    Creates a BacktestRun record (status=queued), fires a background task that:
    1. Loads OHLCV via yfinance (free)
    2. Calls strategy.backtest_signals(df) to get entry/exit signals
    3. Passes signals + prices to run_backtest() in engine.py
    4. Persists BacktestMetrics to BacktestResult

    Poll GET /backtests/{id} for results.
    """
    run = BacktestRun(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        strategy_name=body.strategy_name,
        symbol=body.symbol,
        interval=body.interval,
        start_date=body.start_date,
        end_date=body.end_date,
        params={"initial_equity": body.initial_equity},
        status="queued",
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    await db.commit()

    # Wire actual execution as a background task
    background_tasks.add_task(
        _run_backtest_task,
        run.id,
        body.strategy_name,
        body.symbol,
        body.interval,
        body.start_date,
        body.end_date,
        body.initial_equity,
    )

    fresh = await db.execute(
        select(BacktestRun).where(BacktestRun.id == run.id)
        .options(selectinload(BacktestRun.result))
    )
    return BacktestOut.from_run(fresh.scalar_one())


@router.post("/")
@limiter.limit("5/minute")
async def trigger_backtest(
    request: Request,
    body: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """POST /backtests/ — alias for /backtests/run (backward compatibility)."""
    return await trigger_backtest_run(request, body, background_tasks, db, current_user)


@router.post("/walk-forward")
@limiter.limit("3/minute")
async def trigger_walk_forward(
    request: Request,
    body: WalkForwardRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    POST /backtests/walk-forward — trigger walk-forward validation.

    Rolls a train/test window across the full history using the strategy's
    backtest_signals(). Returns average OOS Sharpe and per-window metrics.

    Requires at least train_years * 252 + test_months * 21 bars of data.
    """
    run = BacktestRun(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        strategy_name=body.strategy_name,
        symbol=body.symbol,
        interval=body.interval,
        start_date=body.start_date,
        end_date=body.end_date,
        params={
            "initial_equity": body.initial_equity,
            "mode": "walk_forward",
            "train_years": body.train_years,
            "test_months": body.test_months,
        },
        status="queued",
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    await db.commit()

    background_tasks.add_task(
        _run_walk_forward_task,
        run.id,
        body.strategy_name,
        body.symbol,
        body.interval,
        body.start_date,
        body.end_date,
        body.train_years,
        body.test_months,
        body.initial_equity,
    )

    fresh = await db.execute(
        select(BacktestRun).where(BacktestRun.id == run.id)
        .options(selectinload(BacktestRun.result))
    )
    return BacktestOut.from_run(fresh.scalar_one())
