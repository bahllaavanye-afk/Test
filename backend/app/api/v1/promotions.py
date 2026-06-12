"""Strategy promotion pipeline API."""
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.promotion import StrategyPromotion
from app.tasks.promotion_criteria import check_criteria, TRANSITION_MAP
from datetime import datetime, timezone

router = APIRouter(prefix="/promotions", tags=["promotions"])


@router.get("/")
async def list_promotions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(StrategyPromotion).order_by(StrategyPromotion.created_at.desc()))
    return [_serialize(p) for p in result.scalars().all()]


@router.get("/{promotion_id}")
async def get_promotion(
    promotion_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    p = await db.get(StrategyPromotion, promotion_id)
    if not p:
        raise HTTPException(status_code=404, detail="Promotion not found")
    return _serialize(p)


class CreatePromotionRequest(BaseModel):
    strategy_id: str
    strategy_name: str
    notes: str | None = None


@router.post("/")
async def create_promotion(
    req: CreatePromotionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Register a strategy in the promotion pipeline (starts at 'paper' stage)."""
    ts = datetime.now(timezone.utc).isoformat()
    p = StrategyPromotion(
        strategy_id=req.strategy_id,
        strategy_name=req.strategy_name,
        current_stage="paper",
        paper_started_at=ts,
        notes=req.notes,
    )
    db.add(p)
    await db.commit()
    await db.refresh(p)
    return _serialize(p)


class UpdateMetricsRequest(BaseModel):
    sharpe: float
    win_rate: float
    max_drawdown: float
    num_trades: int
    days_in_stage: int
    sortino: float | None = None
    extra: dict | None = None


@router.post("/{promotion_id}/metrics")
async def update_metrics(
    promotion_id: str,
    req: UpdateMetricsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update metrics for the current stage."""
    p = await db.get(StrategyPromotion, promotion_id)
    if not p:
        raise HTTPException(status_code=404, detail="Promotion not found")

    metrics = req.model_dump(exclude_none=True)
    if p.current_stage == "paper":
        p.paper_metrics = metrics
    elif p.current_stage == "shadow":
        p.shadow_metrics = metrics
    elif p.current_stage == "staging":
        p.staging_metrics = metrics
    elif p.current_stage == "live":
        p.live_metrics = metrics

    db.add(p)
    await db.commit()
    return _serialize(p)


@router.post("/{promotion_id}/approve")
async def approve_promotion(
    promotion_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Approve strategy for promotion to next stage."""
    p = await db.get(StrategyPromotion, promotion_id)
    if not p:
        raise HTTPException(status_code=404, detail="Promotion not found")
    if not p.promotion_ready:
        raise HTTPException(status_code=400, detail="Strategy has not yet passed promotion criteria")

    old_stage = p.current_stage
    stage_order = ["paper", "shadow", "staging", "live"]
    idx = stage_order.index(old_stage)
    if idx >= len(stage_order) - 1:
        raise HTTPException(status_code=400, detail="Strategy is already at final stage")

    new_stage = stage_order[idx + 1]
    ts = datetime.now(timezone.utc).isoformat()
    p.current_stage = new_stage
    p.promotion_ready = False
    p.awaiting_approval = False
    p.approved_by = current_user.id
    p.approved_at = ts
    setattr(p, f"{new_stage}_started_at", ts)

    # Notify Slack
    from app.notifications.slack import slack
    await slack.notify_system(
        f":white_check_mark: Strategy `{p.strategy_name}` promoted from *{old_stage}* → *{new_stage}* "
        f"by {current_user.email}",
        level="info"
    )

    db.add(p)
    await db.commit()
    return _serialize(p)


class RejectRequest(BaseModel):
    reason: str


@router.post("/{promotion_id}/reject")
async def reject_promotion(
    promotion_id: str,
    req: RejectRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Reject strategy promotion — marks it as rejected."""
    p = await db.get(StrategyPromotion, promotion_id)
    if not p:
        raise HTTPException(status_code=404, detail="Promotion not found")

    old_stage = p.current_stage
    ts = datetime.now(timezone.utc).isoformat()
    p.current_stage = "rejected"
    p.promotion_ready = False
    p.awaiting_approval = False
    p.rejection_reason = req.reason

    history = list(p.review_history or [])
    history.append({"ts": ts, "event": "rejected", "stage": old_stage, "reason": req.reason})
    p.review_history = history

    from app.notifications.slack import slack
    await slack.notify_system(
        f":x: Strategy `{p.strategy_name}` rejected at stage *{old_stage}*. Reason: {req.reason}",
        level="warning"
    )

    db.add(p)
    await db.commit()
    return _serialize(p)


@router.post("/{promotion_id}/review")
async def trigger_review(
    promotion_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Manually trigger a holistic review for this promotion."""
    p = await db.get(StrategyPromotion, promotion_id)
    if not p:
        raise HTTPException(status_code=404, detail="Promotion not found")

    from app.tasks.promotion_criteria import check_criteria, TRANSITION_MAP
    from app.tasks.holistic_review import _get_stage_metrics, _notify_promotion_ready

    transition = TRANSITION_MAP.get(p.current_stage)
    if not transition:
        return {"status": "no_transition", "stage": p.current_stage}

    metrics = _get_stage_metrics(p)
    passed, failures = check_criteria(metrics, transition)

    ts = datetime.now(timezone.utc).isoformat()
    entry = {"ts": ts, "stage": p.current_stage, "transition": transition, "passed": passed, "metrics": metrics, "failures": failures}
    history = list(p.review_history or [])
    history.append(entry)
    p.review_history = history
    p.last_review_at = ts

    if passed and not p.awaiting_approval:
        p.promotion_ready = True
        p.promotion_ready_stage = transition.split("_to_")[1]
        p.awaiting_approval = True
        await _notify_promotion_ready(p, metrics, transition)

    db.add(p)
    await db.commit()

    return {"passed": passed, "failures": failures, "transition": transition, "metrics": metrics}


def _serialize(p: StrategyPromotion) -> dict:
    return {
        "id": p.id,
        "strategy_id": p.strategy_id,
        "strategy_name": p.strategy_name,
        "current_stage": p.current_stage,
        "paper_metrics": p.paper_metrics,
        "shadow_metrics": p.shadow_metrics,
        "staging_metrics": p.staging_metrics,
        "live_metrics": p.live_metrics,
        "paper_started_at": p.paper_started_at,
        "shadow_started_at": p.shadow_started_at,
        "staging_started_at": p.staging_started_at,
        "live_started_at": p.live_started_at,
        "promotion_ready": p.promotion_ready,
        "promotion_ready_stage": p.promotion_ready_stage,
        "awaiting_approval": p.awaiting_approval,
        "approved_by": p.approved_by,
        "approved_at": p.approved_at,
        "rejection_reason": p.rejection_reason,
        "last_review_at": p.last_review_at,
        "review_history": p.review_history,
        "notes": p.notes,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }
