"""Task management and employee dispatch endpoints."""
from datetime import UTC, datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_current_user
from app.database import get_db
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.user import User

router = APIRouter(prefix="/tasks", tags=["tasks"])

EMPLOYEE_TYPES = {
    "strategy_agent": {"display": "Strategy Agent", "skills": ["analyze", "backtest", "optimize", "compare"]},
    "ml_agent": {"display": "ML Agent", "skills": ["retrain", "predict", "evaluate", "feature_engineering"]},
    "risk_agent": {"display": "Risk Agent", "skills": ["risk_check", "circuit_breaker", "portfolio_analysis"]},
    "data_agent": {"display": "Data Agent", "skills": ["fetch_ohlcv", "seed_strategies", "clean_data"]},
    "execution_agent": {"display": "Execution Agent", "skills": ["execute", "twap", "vwap", "slippage_analysis"]},
    "research_agent": {"display": "Research Agent", "skills": ["alpha_mining", "factor_research", "paper_review"]},
}


class TaskCreate(BaseModel):
    title: str
    description: str | None = None
    task_type: str
    assigned_to: str | None = None
    priority: TaskPriority = TaskPriority.medium
    params: dict = {}


class TaskUpdate(BaseModel):
    status: TaskStatus | None = None
    result: dict | None = None
    error_message: str | None = None
    progress_pct: float | None = None
    assigned_to: str | None = None


class TaskOut(BaseModel):
    id: str
    title: str
    description: str | None
    task_type: str
    assigned_to: str | None
    assigned_by: str | None
    status: TaskStatus
    priority: TaskPriority
    params: dict
    result: dict | None
    error_message: str | None
    progress_pct: float
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    model_config = {"from_attributes": True}


@router.get("/employees")
async def list_employees(current_user: User = Depends(get_current_user)):
    """List available employee agents and their skills."""
    return EMPLOYEE_TYPES


@router.post("/", response_model=TaskOut)
async def create_task(
    body: TaskCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    task = Task(
        title=body.title,
        description=body.description,
        task_type=body.task_type,
        assigned_to=body.assigned_to,
        assigned_by=str(current_user.id),
        priority=body.priority,
        params=body.params,
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)
    return task


@router.get("/", response_model=list[TaskOut])
async def list_tasks(
    status: TaskStatus | None = Query(None),
    assigned_to: str | None = Query(None),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = select(Task).order_by(Task.created_at.desc()).limit(limit)
    if status:
        q = q.where(Task.status == status)
    if assigned_to:
        q = q.where(Task.assigned_to == assigned_to)
    result = await db.execute(q)
    return result.scalars().all()


@router.patch("/{task_id}", response_model=TaskOut)
async def update_task(
    task_id: str,
    body: TaskUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(404, "Task not found")
    if body.status is not None:
        task.status = body.status
        if body.status == TaskStatus.running and not task.started_at:
            task.started_at = datetime.now(UTC)
        if body.status in (TaskStatus.done, TaskStatus.failed, TaskStatus.cancelled):
            task.completed_at = datetime.now(UTC)
    if body.result is not None:
        task.result = body.result
    if body.error_message is not None:
        task.error_message = body.error_message
    if body.progress_pct is not None:
        task.progress_pct = body.progress_pct
    if body.assigned_to is not None:
        task.assigned_to = body.assigned_to
    await db.commit()
    await db.refresh(task)
    return task


@router.delete("/{task_id}")
async def delete_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(404, "Task not found")
    await db.delete(task)
    await db.commit()
    return {"deleted": task_id}
