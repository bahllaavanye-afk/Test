"""
Self-Improving Loop — runs every hour via APScheduler.

Cycle:
  1. Pull recent trade performance from DB
  2. Ask free LLM (race mode) for strategy improvement ideas
  3. Score each active strategy (Sharpe, win rate, drawdown)
  4. Auto-disable strategies with Sharpe < 0 over last 30 days
  5. Write observations to AgentMemory for other agents
  6. Broadcast regime + recommendation to Redis pub/sub
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasks.free_llm_router import call_race
from app.tasks.agent_memory import AgentMemory

logger = logging.getLogger(__name__)


class SelfImprovingLoop:
    def __init__(self, db_session_factory: Any, redis_client: Any):
        self._factory = db_session_factory
        self._memory = AgentMemory(redis_client)
        self._redis = redis_client

    async def run_cycle(self) -> None:
        """Run one full self‑improving iteration."""
        logger.info("SelfImprovingLoop: starting hourly cycle")
        try:
            metrics = await self._collect_strategy_metrics()
            await self._auto_disable_underperformers(metrics)
            await self._llm_improvement_pass(metrics)
            await self._broadcast_regime(metrics)
            logger.info(
                "SelfImprovingLoop: cycle complete (%d strategies evaluated)",
                len(metrics) if metrics else 0,
            )
        except Exception as e:
            logger.exception("SelfImprovingLoop cycle error: %s", e)

    # ── Metric collection ─────────────────────────────────────────────────────

    async def _collect_strategy_metrics(self) -> List[dict]:
        """Pull per‑strategy Sharpe + win‑rate from trade history (last 30 d)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        async with self._factory() as session:
            try:
                result = await session.execute(
                    text(
                        """
                SELECT
                    strategy_name,
                    COUNT(*) AS num_trades,
                    SUM(pnl) AS total_pnl,
                    AVG(pnl) AS avg_pnl,
                    STDDEV(pnl) AS std_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::float / COUNT(*) AS win_rate
                FROM trades
                WHERE closed_at >= :cutoff AND strategy_name IS NOT NULL
                GROUP BY strategy_name
                """
                    ),
                    {"cutoff": cutoff},
                )
                rows = result.fetchall()
            except Exception as exc:
                logger.exception("Failed to collect strategy metrics: %s", exc)
                rows = []

        metrics: List[dict] = []
        for row in rows:
            # Guard against division by zero or null std deviation
            std = row.std_pnl if row.std_pnl and row.std_pnl > 0 else 1e-9
            sharpe = (row.avg_pnl / std) * (252 ** 0.5) if std else 0.0
            metrics.append(
                {
                    "strategy": row.strategy_name,
                    "num_trades": row.num_trades,
                    "total_pnl": float(row.total_pnl or 0),
                    "avg_pnl": float(row.avg_pnl or 0),
                    "win_rate": float(row.win_rate or 0),
                    "sharpe": round(sharpe, 3),
                }
            )
        return metrics

    # ── Auto-disable ──────────────────────────────────────────────────────────

    async def _auto_disable_underperformers(self, metrics: Iterable[dict] | None) -> None:
        """Disable strategies with Sharpe < 0 and ≥ 10 trades in the last 30 d."""
        if not metrics:
            return

        underperformers = [
            m
            for m in metrics
            if m.get("sharpe", 0) < 0 and m.get("num_trades", 0) >= 10
        ]
        if not underperformers:
            return

        async with self._factory() as session:
            for m in underperformers:
                await session.execute(
                    text(
                        """
                    UPDATE strategies SET is_active = false, disabled_reason = :reason
                    WHERE name = :name AND is_active = true
                    """
                    ),
                    {
                        "name": m["strategy"],
                        "reason": f"auto-disabled: Sharpe={m['sharpe']:.2f} (30d)",
                    },
                )
            await session.commit()

        names = [m["strategy"] for m in underperformers]
        logger.info("SelfImprovingLoop: auto-disabled %s", names)
        await self._memory.write("auto_disabled", {"strategies": names})

    # ── LLM improvement pass ──────────────────────────────────────────────────

    async def _llm_improvement_pass(self, metrics: Iterable[dict] | None) -> None:
        """Query a free LLM for improvement suggestions based on recent performance."""
        if not metrics:
            return

        # Defensive sorting – ensure we have a list to slice
        metrics_list = list(metrics)
        top = sorted(metrics_list, key=lambda m: m.get("sharpe", 0), reverse=True)[:5]
        bottom = sorted(metrics_list, key=lambda m: m.get("sharpe", 0))[:3]

        prompt = f"""You are a quantitative trading researcher.

Top performing strategies (last 30d):
{json.dumps(top, indent=2)}

Underperforming strategies:
{json.dumps(bottom, indent=2)}

Suggest 3 specific, actionable improvements:
1. Parameter tuning for the worst performer
2. A new indicator combination to test
3. A risk rule change to protect capital

Be concise. Each suggestion under 2 sentences."""
        try:
            response = await call_race(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )
        except Exception as exc:
            logger.exception("LLM call failed: %s", exc)
            response = None

        if response:
            provider = getattr(response, "provider", "unknown")
            suggestion = getattr(response, "content", "")
            await self._memory.write(
                "llm_suggestions",
                {"provider": provider, "suggestion": suggestion},
            )
            logger.info("SelfImprovingLoop: LLM suggestion from %s stored", provider)

    # ── Regime broadcast ──────────────────────────────────────────────────────

    async def _broadcast_regime(self, metrics: Iterable[dict] | None) -> None:
        """Publish current platform regime based on strategy health."""
        if not metrics:
            # No data – treat as neutral side‑ways regime
            health = 0.0
            profitable = 0
            total = 0
        else:
            metrics_list = list(metrics)
            profitable = sum(1 for m in metrics_list if m.get("sharpe", 0) > 0.5)
            total = len(metrics_list)
            health = profitable / total if total else 0.0

        regime = (
            "bull"
            if health > 0.6
            else ("bear" if health < 0.3 else "sideways")
        )
        await self._memory.set_latest(
            "platform_health",
            {
                "regime": regime,
                "health_ratio": round(health, 3),
                "profitable_strategies": profitable,
                "total_strategies": total,
            },
        )

        try:
            await self._redis.publish(
                "platform:regime",
                json.dumps({"regime": regime, "health": health}),
            )
        except Exception:
            # Redis failures should not crash the loop
            logger.debug("Redis publish failed during regime broadcast", exc_info=True)