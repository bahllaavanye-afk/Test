from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

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
        start_time = time.perf_counter()
        logger.info("SelfImprovingLoop: starting hourly cycle")
        try:
            metrics = await self._collect_strategy_metrics()
            collection_time = time.perf_counter()
            self._log_metrics(metrics, collection_time - start_time)

            await self._auto_disable_underperformers(metrics)
            await self._llm_improvement_pass(metrics)
            await self._broadcast_regime(metrics)

            total_time = time.perf_counter() - start_time
            logger.info(
                "SelfImprovingLoop: cycle complete (%d strategies evaluated) in %.3f seconds",
                len(metrics),
                total_time,
            )
        except Exception as e:
            logger.exception("SelfImprovingLoop cycle error: %s", e)

    # ── Metric collection ─────────────────────────────────────────────────────

    async def _collect_strategy_metrics(self) -> list[dict]:
        """Pull per-strategy Sharpe + win-rate from trade history (last 30d)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
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

    def _log_metrics(self, metrics: list[dict], collection_duration: float) -> None:
        """Log aggregated metrics in a structured INFO message."""
        total_strategies = len(metrics)
        total_pnl = sum(m["total_pnl"] for m in metrics)
        avg_sharpe = round(sum(m["sharpe"] for m in metrics) / total_strategies, 3) if total_strategies else 0.0
        logger.info(
            "SelfImprovingLoop: collected metrics - strategies=%d, total_pnl=%.2f, avg_sharpe=%.3f, collection_time=%.3fs",
            total_strategies,
            total_pnl,
            avg_sharpe,
            collection_duration,
        )

    # ── Auto-disable ──────────────────────────────────────────────────────────

    async def _auto_disable_underperformers(self, metrics: list[dict]) -> None:
        """Disable strategies with Sharpe < 0 and >= 10 trades in the last 30 days."""
        underperformers = [m for m in metrics if m["sharpe"] < 0 and m["num_trades"] >= 10]
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

    async def _llm_improvement_pass(self, metrics: list[dict]) -> None:
        if not metrics:
            return

        top = sorted(metrics, key=lambda m: m["sharpe"], reverse=True)[:5]
        bottom = sorted(metrics, key=lambda m: m["sharpe"])[:3]

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

        response = await call_race(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=512,
        )
        if response:
            await self._memory.write(
                "llm_suggestions",
                {
                    "provider": response.provider,
                    "suggestion": response.content,
                },
            )
            logger.info("SelfImprovingLoop: LLM suggestion from %s stored", response.provider)

    # ── Regime broadcast ──────────────────────────────────────────────────────

    async def _broadcast_regime(self, metrics: list[dict]) -> None:
        profitable = sum(1 for m in metrics if m["sharpe"] > 0.5)
        total = len(metrics) or 1
        health = profitable / total

        regime = "bull" if health > 0.6 else ("bear" if health < 0.3 else "sideways")
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
            await self._redis.publish("platform:regime", json.dumps({"regime": regime, "health": health}))
        except Exception as exc:  # noqa: BLE001 — subscribers just miss one regime tick
            logger.debug("self-improving loop: regime publish failed: %s", exc)