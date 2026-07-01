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
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasks.free_llm_router import call_race
from app.tasks.agent_memory import AgentMemory

logger = logging.getLogger(__name__)

# Constants
METRICS_LOOKBACK_DAYS: int = 30
SHARPE_DISABLE_THRESHOLD: float = 0.0
MIN_TRADES_FOR_DISABLE: int = 10
TOP_STRATEGIES_COUNT: int = 5
BOTTOM_STRATEGIES_COUNT: int = 3
PROFITABLE_SHARPE_THRESHOLD: float = 0.5
BULL_HEALTH_THRESHOLD: float = 0.6
BEAR_HEALTH_THRESHOLD: float = 0.3
LLM_TEMPERATURE: float = 0.4
LLM_MAX_TOKENS: int = 512
REDIS_REGIME_CHANNEL: str = "platform:regime"
MEMORY_KEY_AUTO_DISABLED: str = "auto_disabled"
MEMORY_KEY_LLM_SUGGESTIONS: str = "llm_suggestions"
MEMORY_KEY_PLATFORM_HEALTH: str = "platform_health"
DISABLE_REASON_TEMPLATE: str = "auto-disabled: Sharpe={sharpe:.2f} (30d)"


class SelfImprovingLoop:
    def __init__(self, db_session_factory: Any, redis_client: Any):
        self._factory = db_session_factory
        self._memory = AgentMemory(redis_client)
        self._redis = redis_client

    async def run_cycle(self) -> None:
        logger.info("SelfImprovingLoop: starting hourly cycle")
        try:
            metrics = await self._collect_strategy_metrics()
            await self._auto_disable_underperformers(metrics)
            await self._llm_improvement_pass(metrics)
            await self._broadcast_regime(metrics)
            logger.info("SelfImprovingLoop: cycle complete (%d strategies evaluated)", len(metrics))
        except Exception as e:
            logger.exception("SelfImprovingLoop cycle error: %s", e)

    # ── Metric collection ─────────────────────────────────────────────────────

    async def _collect_strategy_metrics(self) -> list[dict]:
        """Pull per-strategy Sharpe + win-rate from trade history (last 30d)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=METRICS_LOOKBACK_DAYS)
        async with self._factory() as session:
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

        metrics = []
        for row in rows:
            std = row.std_pnl or 1e-9
            sharpe = (row.avg_pnl / std) * (252 ** 0.5) if std > 0 else 0
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

    async def _auto_disable_underperformers(self, metrics: list[dict]) -> None:
        """Disable strategies with Sharpe < 0 and >= 10 trades in the last 30 days."""
        underperformers = [
            m
            for m in metrics
            if m["sharpe"] < SHARPE_DISABLE_THRESHOLD and m["num_trades"] >= MIN_TRADES_FOR_DISABLE
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
                        "reason": DISABLE_REASON_TEMPLATE.format(sharpe=m["sharpe"]),
                    },
                )
            await session.commit()

        names = [m["strategy"] for m in underperformers]
        logger.info("SelfImprovingLoop: auto-disabled %s", names)
        await self._memory.write(MEMORY_KEY_AUTO_DISABLED, {"strategies": names})

    # ── LLM improvement pass ──────────────────────────────────────────────────

    async def _llm_improvement_pass(self, metrics: list[dict]) -> None:
        if not metrics:
            return

        top = sorted(metrics, key=lambda m: m["sharpe"], reverse=True)[:TOP_STRATEGIES_COUNT]
        bottom = sorted(metrics, key=lambda m: m["sharpe"])[:BOTTOM_STRATEGIES_COUNT]

        prompt = f"""You are a quantitative trading researcher.

Top performing strategies (last {METRICS_LOOKBACK_DAYS}d):
{json.dumps(top, indent=2)}

Underperforming strategies:
{json.dumps(bottom, indent=2)}

Suggest 3 specific, actionable improvements:
1. Parameter tuning for the worst performer
2. A new indicator combination to test
3. A risk rule change to protect capital

Be concise. Each suggestion under 2 sentences."""

        response = await call_race(
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        if response:
            await self._memory.write(
                MEMORY_KEY_LLM_SUGGESTIONS,
                {"provider": response.provider, "suggestion": response.content},
            )
            logger.info("SelfImprovingLoop: LLM suggestion from %s stored", response.provider)

    # ── Regime broadcast ──────────────────────────────────────────────────────

    async def _broadcast_regime(self, metrics: list[dict]) -> None:
        profitable = sum(1 for m in metrics if m["sharpe"] > PROFITABLE_SHARPE_THRESHOLD)
        total = len(metrics) or 1
        health = profitable / total

        if health > BULL_HEALTH_THRESHOLD:
            regime = "bull"
        elif health < BEAR_HEALTH_THRESHOLD:
            regime = "bear"
        else:
            regime = "sideways"

        await self._memory.set_latest(
            MEMORY_KEY_PLATFORM_HEALTH,
            {
                "regime": regime,
                "health_ratio": round(health, 3),
                "profitable_strategies": profitable,
                "total_strategies": total,
            },
        )

        try:
            await self._redis.publish(
                REDIS_REGIME_CHANNEL, json.dumps({"regime": regime, "health": health})
            )
        except Exception:
            pass