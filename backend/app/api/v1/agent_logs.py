"""Agent activity log REST endpoints."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_superuser, get_current_user
from app.database import get_db
from app.models.agent_log import AgentActivityLog
from app.models.user import User

router = APIRouter(prefix="/agent-logs", tags=["agent-logs"])


class ReviewRequest(BaseModel):
    note: str
    resolved: bool


@router.get("/")
async def list_agent_logs(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    employee_id: str | None = Query(None),
    agent_type: str | None = Query(None),
    status: str | None = Query(None),
    anomaly_only: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    stmt = select(AgentActivityLog).order_by(AgentActivityLog.created_at.desc())

    if employee_id:
        stmt = stmt.where(AgentActivityLog.employee_id == employee_id)
    if agent_type:
        stmt = stmt.where(AgentActivityLog.agent_type == agent_type)
    if status:
        stmt = stmt.where(AgentActivityLog.status == status)
    if anomaly_only:
        stmt = stmt.where(AgentActivityLog.is_anomaly == True)  # noqa: E712

    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "id": r.id,
            "employee_id": r.employee_id,
            "agent_type": r.agent_type,
            "action": r.action,
            "tool_used": r.tool_used,
            "input_summary": r.input_summary,
            "output_summary": r.output_summary,
            "duration_ms": r.duration_ms,
            "status": r.status,
            "error_message": r.error_message,
            "anomaly_score": r.anomaly_score,
            "is_anomaly": r.is_anomaly,
            "reviewed_by": r.reviewed_by,
            "reviewed_at": r.reviewed_at,
            "review_note": r.review_note,
            "strategy_name": r.strategy_name,
            "symbol": r.symbol,
            "account_id": r.account_id,
            "extra": r.extra,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.get("/stats")
async def agent_log_stats(
    hours: int = Query(24, ge=1, le=168),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return counts by employee, by action, by status, anomaly count, etc."""
    since = datetime.now(UTC) - timedelta(hours=hours)

    # Total count
    total_stmt = select(func.count(AgentActivityLog.id)).where(
        AgentActivityLog.created_at >= since
    )
    total_result = await db.execute(total_stmt)
    total = total_result.scalar() or 0

    # Error count
    error_stmt = select(func.count(AgentActivityLog.id)).where(
        AgentActivityLog.created_at >= since,
        AgentActivityLog.status == "error",
    )
    error_result = await db.execute(error_stmt)
    error_count = error_result.scalar() or 0

    # Anomaly count
    anomaly_stmt = select(func.count(AgentActivityLog.id)).where(
        AgentActivityLog.created_at >= since,
        AgentActivityLog.is_anomaly == True,  # noqa: E712
    )
    anomaly_result = await db.execute(anomaly_stmt)
    anomaly_count = anomaly_result.scalar() or 0

    # Counts by employee
    by_employee_stmt = (
        select(AgentActivityLog.employee_id, func.count(AgentActivityLog.id))
        .where(AgentActivityLog.created_at >= since)
        .group_by(AgentActivityLog.employee_id)
        .order_by(func.count(AgentActivityLog.id).desc())
        .limit(20)
    )
    by_employee_result = await db.execute(by_employee_stmt)
    by_employee = {row[0] or "unknown": row[1] for row in by_employee_result.all()}

    # Counts by action
    by_action_stmt = (
        select(AgentActivityLog.action, func.count(AgentActivityLog.id))
        .where(AgentActivityLog.created_at >= since)
        .group_by(AgentActivityLog.action)
        .order_by(func.count(AgentActivityLog.id).desc())
        .limit(20)
    )
    by_action_result = await db.execute(by_action_stmt)
    by_action = {row[0]: row[1] for row in by_action_result.all()}

    # Counts by status
    by_status_stmt = (
        select(AgentActivityLog.status, func.count(AgentActivityLog.id))
        .where(AgentActivityLog.created_at >= since)
        .group_by(AgentActivityLog.status)
    )
    by_status_result = await db.execute(by_status_stmt)
    by_status = {row[0]: row[1] for row in by_status_result.all()}

    # Active employees (distinct employee_ids in last hour)
    active_stmt = select(func.count(func.distinct(AgentActivityLog.employee_id))).where(
        AgentActivityLog.created_at >= datetime.now(UTC) - timedelta(hours=1)
    )
    active_result = await db.execute(active_stmt)
    active_employees = active_result.scalar() or 0

    return {
        "hours": hours,
        "total_actions": total,
        "error_count": error_count,
        "anomaly_count": anomaly_count,
        "active_employees": active_employees,
        "by_employee": by_employee,
        "by_action": by_action,
        "by_status": by_status,
    }


@router.post("/{log_id}/review")
async def review_log(
    log_id: str,
    body: ReviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_superuser),
) -> dict:
    result = await db.execute(
        select(AgentActivityLog).where(AgentActivityLog.id == log_id)
    )
    entry = result.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="Log entry not found")

    entry.reviewed_by = current_user.email or current_user.id
    entry.reviewed_at = datetime.now(UTC).isoformat()
    entry.review_note = body.note
    if body.resolved:
        entry.is_anomaly = False

    await db.commit()
    await db.refresh(entry)

    return {
        "id": entry.id,
        "reviewed_by": entry.reviewed_by,
        "reviewed_at": entry.reviewed_at,
        "review_note": entry.review_note,
        "is_anomaly": entry.is_anomaly,
    }
