"""Backtest trigger and result retrieval endpoints."""
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.backtest import BacktestRun
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


class BacktestOut(BaseModel):
    id: str
    strategy_name: str
    symbol: str
    interval: str
    status: str
    sharpe: float | None = None
    max_drawdown: float | None = None
    total_return: float | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_run(cls, run) -> "BacktestOut":
        result = run.result
        return cls(
            id=run.id, strategy_name=run.strategy_name, symbol=run.symbol,
            interval=run.interval, status=run.status, created_at=run.created_at,
            sharpe=result.sharpe_ratio if result else None,
            max_drawdown=result.max_drawdown if result else None,
            total_return=result.total_return if result else None,
        )


@router.get("/")
async def list_backtests(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(BacktestRun).where(BacktestRun.user_id == current_user.id)
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
    await db.refresh(run)
    return BacktestOut.from_run(run)


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
