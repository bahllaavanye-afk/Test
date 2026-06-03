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


@router.get("/")
async def list_backtests(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(BacktestRun).where(BacktestRun.user_id == current_user.id)
        .options(selectinload(BacktestRun.result))
        .order_by(BacktestRun.created_at.desc()).limit(20)
    )
    runs = result.scalars().all()
    return [BacktestOut.from_run(r) for r in runs]


@router.post("/")
async def trigger_backtest(
    body: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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
    # Use explicit query to avoid lazy-load issue on async session
    fresh = await db.execute(
        select(BacktestRun).where(BacktestRun.id == run.id)
        .options(selectinload(BacktestRun.result))
    )
    return BacktestOut.from_run(fresh.scalar_one())


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
