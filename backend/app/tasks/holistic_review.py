"""Holistic strategy review — runs daily at 06:00 UTC."""
from __future__ import annotations

from datetime import UTC, datetime

from app.utils.logging import logger


async def run_holistic_review(db_session_factory=None) -> None:
    """Review all active strategy promotions and fire Slack alerts for promotion-ready ones."""
    from sqlalchemy import select

    from app.models.promotion import StrategyPromotion
    from app.tasks.promotion_criteria import TRANSITION_MAP, check_criteria

    if db_session_factory is None:
        from app.database import AsyncSessionLocal as db_session_factory

    try:
        # Collect Slack payloads here; notifications fired AFTER commit to prevent
        # a Slack outage from rolling back all promotion_ready mutations.
        pending_notifications: list[tuple[str, str, dict, str]] = []

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
                # Cap history to last 90 entries to avoid unbounded growth
                p.review_history = history[-90:]
                p.last_review_at = ts

                if passed and not p.awaiting_approval:
                    p.promotion_ready = True
                    p.promotion_ready_stage = transition.split("_to_")[1]
                    p.awaiting_approval = True
                    promoted_count += 1
                    pending_notifications.append(
                        (p.strategy_name, p.current_stage, metrics, transition)
                    )
                elif not passed and p.awaiting_approval:
                    # Criteria degraded after becoming ready — clear the flag
                    p.promotion_ready = False
                    p.awaiting_approval = False

                session.add(p)

            await session.commit()
            logger.info("Holistic review complete", reviewed=len(promotions), promotion_ready=promoted_count)

        # Fire Slack notifications outside the session; failures don't affect DB state
        for strategy_name, current_stage, metrics, transition in pending_notifications:
            try:
                await _notify_promotion_ready_simple(strategy_name, current_stage, metrics, transition)
            except Exception as notify_err:
                logger.warning("Slack notification failed", error=str(notify_err), strategy=strategy_name)

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
    """Kept for backward compatibility with the manual /review endpoint."""
    await _notify_promotion_ready_simple(
        promotion.strategy_name, promotion.current_stage, metrics, transition
    )


async def _notify_promotion_ready_simple(
    strategy_name: str, current_stage: str, metrics: dict, transition: str
) -> None:
    from app.notifications.slack import slack
    next_stage = transition.split("_to_")[1]
    sharpe_val = metrics.get("sharpe", 0)
    sharpe_str = f"{sharpe_val:.2f}" if isinstance(sharpe_val, (int, float)) else str(sharpe_val)
    msg = (
        f":rocket: *Strategy Promotion Ready*\n"
        f"Strategy `{strategy_name}` has passed all criteria "
        f"for promotion from *{current_stage}* → *{next_stage}*.\n\n"
        f"*Current Metrics:*\n"
        f"• Sharpe: `{sharpe_str}`\n"
        f"• Win Rate: `{metrics.get('win_rate', 0):.1%}`\n"
        f"• Max Drawdown: `{metrics.get('max_drawdown', 0):.1%}`\n"
        f"• Days in stage: `{metrics.get('days_in_stage', 0)}`\n"
        f"• Trades: `{metrics.get('num_trades', 0)}`\n\n"
        f"*Action required:* Review at `/risk-manager`"
    )
    await slack.notify_system(msg, level="info")
