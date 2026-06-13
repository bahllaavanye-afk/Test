"""
Syncs live paper-trading performance into the strategy promotion pipeline.
Runs every 6 hours via the scheduler.

For each active promotion, queries closed Trade records for that strategy
and computes Sharpe, win rate, max drawdown, and days_in_stage, then
calls POST /promotions/{id}/metrics internally to update the record.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from app.utils.logging import logger


async def sync_promotion_metrics(db_session_factory=None) -> int:
    """Update metrics for all active (non-live, non-rejected) promotions. Returns count updated."""
    from sqlalchemy import select
    from app.models.promotion import StrategyPromotion
    from app.models.trade import Trade

    if db_session_factory is None:
        from app.database import AsyncSessionLocal as db_session_factory

    updated = 0
    try:
        async with db_session_factory() as session:
            result = await session.execute(
                select(StrategyPromotion).where(
                    StrategyPromotion.current_stage.in_(["paper", "shadow", "staging"])
                )
            )
            promotions = result.scalars().all()

            for p in promotions:
                stage_start = _parse_ts(
                    p.paper_started_at if p.current_stage == "paper"
                    else p.shadow_started_at if p.current_stage == "shadow"
                    else p.staging_started_at
                )
                if stage_start is None:
                    continue

                trade_result = await session.execute(
                    select(Trade).where(
                        Trade.strategy_name == p.strategy_name,
                        Trade.closed_at >= stage_start,
                    )
                )
                trades = trade_result.scalars().all()

                if len(trades) < 3:
                    continue

                metrics = _compute_metrics(trades, stage_start)
                current = _get_current_metrics(p)

                if _significantly_different(current, metrics):
                    _set_stage_metrics(p, metrics)
                    session.add(p)
                    updated += 1

            await session.commit()
    except Exception as e:
        logger.error("sync_promotion_metrics failed", error=str(e))

    logger.info("Promotion metrics synced", updated=updated)
    return updated


def _parse_ts(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _compute_metrics(trades, stage_start: datetime) -> dict:
    """Compute Sharpe, win_rate, max_drawdown, num_trades, days_in_stage from Trade list."""
    now = datetime.now(timezone.utc)
    days = max(1, (now - stage_start).days)

    pnls = [float(t.realized_pnl) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) if pnls else 0.0

    # Daily aggregated PnL and notional (to compute percentage returns for Sharpe)
    from collections import defaultdict
    daily_pnl: dict[str, float] = defaultdict(float)
    daily_notional: dict[str, float] = defaultdict(float)
    for t in trades:
        day = t.closed_at.strftime("%Y-%m-%d")
        daily_pnl[day] += float(t.realized_pnl)
        daily_notional[day] += float(t.entry_price) * float(t.quantity)

    # Sort by calendar date so the drawdown loop traverses in chronological order
    sorted_days = sorted(daily_pnl.keys())
    # Percentage return per day normalised by notional (independent of position sizing)
    daily_vals = [
        daily_pnl[d] / max(daily_notional[d], 1e-9)
        for d in sorted_days
    ]

    if len(daily_vals) >= 2:
        mean = sum(daily_vals) / len(daily_vals)
        variance = sum((x - mean) ** 2 for x in daily_vals) / (len(daily_vals) - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-9
        sharpe = round((mean / std) * math.sqrt(252), 4)
    else:
        sharpe = 0.0

    # Max drawdown from cumulative percentage returns (chronological order guaranteed above)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in daily_vals:
        cum += v
        if cum > peak:
            peak = cum
        dd = (cum - peak) / max(abs(peak), 1e-9)
        if dd < max_dd:
            max_dd = dd

    # One-tailed t-test: H0 = mean return == 0; need >= 20 days for statistical power
    p_value: float | None = None
    if len(daily_vals) >= 20:
        try:
            from scipy import stats as scipy_stats
            t_stat, two_tail_p = scipy_stats.ttest_1samp(daily_vals, popmean=0)
            p_value = round(float(two_tail_p / 2 if t_stat > 0 else 1.0), 6)
        except ImportError:
            pass  # scipy not installed — leave p_value as None

    return {
        "sharpe": sharpe,
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_dd, 4),
        "num_trades": len(trades),
        "days_in_stage": days,
        "p_value": p_value,
    }


def _get_current_metrics(p) -> dict:
    if p.current_stage == "paper":
        return dict(p.paper_metrics or {})
    elif p.current_stage == "shadow":
        return dict(p.shadow_metrics or {})
    elif p.current_stage == "staging":
        return dict(p.staging_metrics or {})
    return {}


def _set_stage_metrics(p, metrics: dict) -> None:
    if p.current_stage == "paper":
        p.paper_metrics = metrics
    elif p.current_stage == "shadow":
        p.shadow_metrics = metrics
    elif p.current_stage == "staging":
        p.staging_metrics = metrics


def _significantly_different(old: dict, new: dict) -> bool:
    """Return True if any metric changed by more than 1%."""
    if not old:
        return True
    for key in ("sharpe", "win_rate", "max_drawdown", "num_trades", "days_in_stage"):
        ov = old.get(key, 0) or 0
        nv = new.get(key, 0) or 0
        if abs(nv - ov) > max(abs(ov) * 0.01, 0.001):
            return True
    return False
