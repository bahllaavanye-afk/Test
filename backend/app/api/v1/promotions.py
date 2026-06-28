"""Strategy promotion pipeline API."""
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_active_superuser, get_current_user, get_db
from app.models.promotion import StrategyPromotion
from app.models.user import User
from app.tasks.promotion_criteria import TRANSITION_MAP, check_criteria

router = APIRouter(prefix="/promotions", tags=["promotions"])


@router.get("/")
async def list_promotions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    result = await db.execute(
        select(StrategyPromotion).order_by(StrategyPromotion.created_at.desc())
    )
    promotions = result.scalars().all()
    if not promotions:
        return []
    return [_serialize(p) for p in promotions]


@router.get("/criteria/all")
async def get_all_criteria(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Dict[str, Any]]:
    """Return promotion criteria thresholds for all transitions."""
    from app.tasks.promotion_criteria import CRITERIA

    if not CRITERIA:
        return {}
    return {
        name: {
            "min_days": c.min_days,
            "min_sharpe": c.min_sharpe,
            "min_win_rate": c.min_win_rate,
            "max_drawdown": c.max_drawdown,
            "min_trades": c.min_trades,
            "require_p_value": c.require_p_value,
        }
        for name, c in CRITERIA.items()
    }


@router.get("/{promotion_id}")
async def get_promotion(
    promotion_id: Optional[str],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not promotion_id:
        raise HTTPException(status_code=400, detail="Promotion ID must be provided")
    p = await db.get(StrategyPromotion, promotion_id)
    if not p:
        raise HTTPException(status_code=404, detail="Promotion not found")
    return _serialize(p)


class CreatePromotionRequest(BaseModel):
    strategy_id: str
    strategy_name: str
    notes: Optional[str] = None

    @validator("strategy_id", "strategy_name")
    def non_empty(cls, v: str, field):
        if not v or not v.strip():
            raise ValueError(f"{field.name} cannot be empty")
        return v.strip()


@router.post("/")
async def create_promotion(
    req: CreatePromotionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Register a strategy in the promotion pipeline (starts at 'paper' stage)."""
    ts = datetime.now(UTC).isoformat()
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
    sortino: Optional[float] = None
    p_value: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


@router.post("/{promotion_id}/metrics")
async def update_metrics(
    promotion_id: Optional[str],
    req: UpdateMetricsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update metrics for the current stage."""
    if not promotion_id:
        raise HTTPException(status_code=400, detail="Promotion ID must be provided")
    p = await db.get(StrategyPromotion, promotion_id)
    if not p:
        raise HTTPException(status_code=404, detail="Promotion not found")

    metrics = req.model_dump(exclude_none=True)
    if not metrics:
        raise HTTPException(status_code=400, detail="No metrics provided")

    if p.current_stage == "paper":
        p.paper_metrics = metrics
    elif p.current_stage == "shadow":
        p.shadow_metrics = metrics
    elif p.current_stage == "staging":
        p.staging_metrics = metrics
    elif p.current_stage == "live":
        p.live_metrics = metrics
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update metrics for unknown stage '{p.current_stage}'",
        )

    db.add(p)
    await db.commit()
    return _serialize(p)


@router.post("/{promotion_id}/approve")
async def approve_promotion(
    promotion_id: Optional[str],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_superuser),
):
    """Approve strategy for promotion to next stage."""
    if not promotion_id:
        raise HTTPException(status_code=400, detail="Promotion ID must be provided")
    p = await db.get(StrategyPromotion, promotion_id)
    if not p:
        raise HTTPException(status_code=404, detail="Promotion not found")
    if not p.promotion_ready:
        raise HTTPException(
            status_code=400, detail="Strategy has not yet passed promotion criteria"
        )

    old_stage = p.current_stage
    stage_order = ["paper", "shadow", "staging", "live"]
    if old_stage not in stage_order:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve: strategy is in stage '{old_stage}'",
        )
    idx = stage_order.index(old_stage)
    if idx >= len(stage_order) - 1:
        raise HTTPException(status_code=400, detail="Strategy is already at final stage")

    new_stage = stage_order[idx + 1]
    ts = datetime.now(UTC).isoformat()
    p.current_stage = new_stage
    p.promotion_ready = False
    p.awaiting_approval = False
    p.approved_by = current_user.id
    p.approved_at = ts
    setattr(p, f"{new_stage}_started_at", ts)

    db.add(p)
    await db.commit()

    # Notify after commit so a Slack outage doesn't roll back the promotion
    try:
        from app.notifications.slack import slack

        await slack.notify_system(
            f":white_check_mark: Strategy `{p.strategy_name}` promoted from *{old_stage}* → *{new_stage}* "
            f"by {current_user.email}",
            level="info",
        )
    except Exception:
        pass

    return _serialize(p)


class RejectRequest(BaseModel):
    reason: str

    @validator("reason")
    def non_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("Rejection reason cannot be empty")
        return v.strip()


@router.post("/{promotion_id}/reject")
async def reject_promotion(
    promotion_id: Optional[str],
    req: RejectRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_superuser),
):
    """Reject strategy promotion — marks it as rejected."""
    if not promotion_id:
        raise HTTPException(status_code=400, detail="Promotion ID must be provided")
    p = await db.get(StrategyPromotion, promotion_id)
    if not p:
        raise HTTPException(status_code=404, detail="Promotion not found")

    old_stage = p.current_stage
    ts = datetime.now(UTC).isoformat()
    p.current_stage = "rejected"
    p.promotion_ready = False
    p.awaiting_approval = False
    p.rejection_reason = req.reason

    history = list(p.review_history or [])
    history.append(
        {"ts": ts, "event": "rejected", "stage": old_stage, "reason": req.reason}
    )
    p.review_history = history

    db.add(p)
    await db.commit()

    try:
        from app.notifications.slack import slack

        await slack.notify_system(
            f":x: Strategy `{p.strategy_name}` rejected at stage *{old_stage}*. Reason: {req.reason}",
            level="warning",
        )
    except Exception:
        pass

    return _serialize(p)


@router.post("/{promotion_id}/review")
async def trigger_review(
    promotion_id: Optional[str],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Manually trigger a holistic review for this promotion."""
    if not promotion_id:
        raise HTTPException(status_code=400, detail="Promotion ID must be provided")
    p = await db.get(StrategyPromotion, promotion_id)
    if not p:
        raise HTTPException(status_code=404, detail="Promotion not found")

    from app.tasks.holistic_review import _get_stage_metrics, _notify_promotion_ready

    transition = TRANSITION_MAP.get(p.current_stage)
    if not transition:
        return {"status": "no_transition", "stage": p.current_stage}

    metrics = _get_stage_metrics(p)
    passed, failures = check_criteria(metrics, transition)

    ts = datetime.now(UTC).isoformat()
    entry = {
        "ts": ts,
        "stage": p.current_stage,
        "transition": transition,
        "passed": passed,
        "metrics": metrics,
        "failures": failures,
    }
    history = list(p.review_history or [])
    history.append(entry)
    p.review_history = history
    p.last_review_at = ts

    if passed and not p.awaiting_approval:
        p.promotion_ready = True
        # transition strings are like "paper_to_shadow"
        p.promotion_ready_stage = transition.split("_to_")[1]
        p.awaiting_approval = True
        await _notify_promotion_ready(p, metrics, transition)

    db.add(p)
    await db.commit()

    return {
        "passed": passed,
        "failures": failures,
        "transition": transition,
        "metrics": metrics,
    }


def _serialize(p: StrategyPromotion) -> Dict[str, Any]:
    """Safely serialize a StrategyPromotion instance to a plain dict."""
    return {
        "id": getattr(p, "id", None),
        "strategy_id": getattr(p, "strategy_id", None),
        "strategy_name": getattr(p, "strategy_name", None),
        "current_stage": getattr(p, "current_stage", None),
        "paper_metrics": getattr(p, "paper_metrics", None),
        "shadow_metrics": getattr(p, "shadow_metrics", None),
        "staging_metrics": getattr(p, "staging_metrics", None),
        "live_metrics": getattr(p, "live_metrics", None),
        "notes": getattr(p, "notes", None),
        "created_at": getattr(p, "created_at", None),
        "paper_started_at": getattr(p, "paper_started_at", None),
        "shadow_started_at": getattr(p, "shadow_started_at", None),
        "staging_started_at": getattr(p, "staging_started_at", None),
        "live_started_at": getattr(p, "live_started_at", None),
        "promotion_ready": getattr(p, "promotion_ready", False),
        "awaiting_approval": getattr(p, "awaiting_approval", False),
        "approved_by": getattr(p, "approved_by", None),
        "approved_at": getattr(p, "approved_at", None),
        "rejection_reason": getattr(p, "rejection_reason", None),
        "review_history": getattr(p, "review_history", []),
        "last_review_at": getattr(p, "last_review_at", None),
    }