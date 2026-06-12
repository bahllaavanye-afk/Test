"""Holistic strategy review — runs daily at 06:00 UTC."""
from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from app.utils.logging import logger


async def run_holistic_review(db_session_factory=None) -> None:
    """Review all active strategy promotions and fire Slack alerts for promotion-ready ones."""
    from sqlalchemy import select
    from app.models.promotion import StrategyPromotion
    from app.tasks.promotion_criteria import check_criteria, TRANSITION_MAP
    from app.notifications.slack import slack

    if db_session_factory is None:
        from app.database import AsyncSessionLocal as db_session_factory

    try:
        async with db_session_factory() as session:
            result = await session.execute(
                select(StrategyPromotion).where(
                    StrategyPromotion.current_stage.in_(["paper", "shadow", "staging"])
                )
            )
            promotions = result.scalars().all()

            promoted_count = 0
            for p in promotions:
                transition = TRANSITION_MAP.get(p.current_stage)
                if not transition:
                    continue

                # Get metrics for current stage
                metrics = _get_stage_metrics(p)
                passed, failures = check_criteria(metrics, transition)

                ts = datetime.now(timezone.utc).isoformat()
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
                    p.promotion_ready_stage = transition.split("_to_")[1]
                    p.awaiting_approval = True
                    promoted_count += 1

                    # Notify Slack
                    await _notify_promotion_ready(p, metrics, transition)

                session.add(p)

            await session.commit()
            logger.info("Holistic review complete", reviewed=len(promotions), promotion_ready=promoted_count)
    except Exception as e:
        logger.error("Holistic review failed", error=str(e))


def _get_stage_metrics(promotion) -> dict:
    """Get metrics dict for the promotion's current stage.

    TODO: Replace mock fallback with real TradeRecord history query once
    trade records are linked to strategy promotions.
    """
    stage = promotion.current_stage
    if stage == "paper":
        return dict(promotion.paper_metrics or {})
    elif stage == "shadow":
        return dict(promotion.shadow_metrics or {})
    elif stage == "staging":
        return dict(promotion.staging_metrics or {})
    return {}


async def _notify_promotion_ready(promotion, metrics: dict, transition: str) -> None:
    from app.notifications.slack import slack
    next_stage = transition.split("_to_")[1]
    msg = (
        f":rocket: *Strategy Promotion Ready*\n"
        f"Strategy `{promotion.strategy_name}` has passed all criteria "
        f"for promotion from *{promotion.current_stage}* → *{next_stage}*.\n\n"
        f"*Current Metrics:*\n"
        f"• Sharpe: `{metrics.get('sharpe', 'N/A') if isinstance(metrics.get('sharpe'), str) else f\"{metrics.get('sharpe', 0):.2f}\"}`\n"
        f"• Win Rate: `{metrics.get('win_rate', 0):.1%}`\n"
        f"• Max Drawdown: `{metrics.get('max_drawdown', 0):.1%}`\n"
        f"• Days in stage: `{metrics.get('days_in_stage', 0)}`\n"
        f"• Trades: `{metrics.get('num_trades', 0)}`\n\n"
        f"*Action required:* Approve via `POST /api/v1/promotions/{promotion.id}/approve` or review at `/risk-manager`"
    )
    await slack.notify_system(msg, level="info")
