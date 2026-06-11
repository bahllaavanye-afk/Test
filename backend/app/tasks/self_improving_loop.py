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

from sqlalchemy import case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.trade import Trade
from app.models.strategy import Strategy
from app.tasks.free_llm_router import call_race
from app.tasks.agent_memory import AgentMemory

logger = logging.getLogger(__name__)


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
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        async with self._factory() as session:
            result = await session.execute(
                select(
                    Trade.strategy_name,
                    func.count(Trade.id).label("num_trades"),
                    func.sum(Trade.realized_pnl).label("total_pnl"),
                    func.avg(Trade.realized_pnl).label("avg_pnl"),
                    func.stddev(Trade.realized_pnl).label("std_pnl"),
                    (
                        func.cast(
                            func.sum(case((Trade.realized_pnl > 0, 1), else_=0)),
                            func.avg(Trade.realized_pnl).type,
                        ) / func.count(Trade.id)
                    ).label("win_rate"),
                )
                .where(
                    Trade.closed_at >= cutoff,
                    Trade.strategy_name.isnot(None),
                )
                .group_by(Trade.strategy_name)
            )
            rows = result.fetchall()

        metrics = []
        for row in rows:
            std = float(row.std_pnl or 0) or 1e-9
            avg = float(row.avg_pnl or 0)
            sharpe = (avg / std) * (252 ** 0.5) if std > 0 else 0
            metrics.append({
                "strategy": row.strategy_name,
                "num_trades": row.num_trades,
                "total_pnl": float(row.total_pnl or 0),
                "avg_pnl": avg,
                "win_rate": float(row.win_rate or 0),
                "sharpe": round(sharpe, 3),
            })
        return metrics

    # ── Auto-disable ──────────────────────────────────────────────────────────

    async def _auto_disable_underperformers(self, metrics: list[dict]) -> None:
        """Disable strategies with Sharpe < 0 and >= 10 trades in the last 30 days.

        Minimum 14-day paper period is enforced: strategies created less than 14 days
        ago are never auto-disabled regardless of Sharpe.
        """
        cutoff_age = datetime.now(timezone.utc) - timedelta(days=14)
        underperformers = [m for m in metrics if m["sharpe"] < 0 and m["num_trades"] >= 10]
        if not underperformers:
            return

        names_to_disable: list[str] = []
        audit_entries: list[dict] = []
        async with self._factory() as session:
            for m in underperformers:
                # Enforce paper-first policy: skip strategies created less than 14 days ago
                strat_result = await session.execute(
                    select(Strategy).where(
                        Strategy.name == m["strategy"],
                        Strategy.is_enabled.is_(True),
                        Strategy.created_at <= cutoff_age,
                    )
                )
                strategy = strat_result.scalar_one_or_none()
                if strategy is None:
                    continue
                await session.execute(
                    update(Strategy)
                    .where(Strategy.name == m["strategy"], Strategy.is_enabled.is_(True))
                    .values(is_enabled=False)
                )
                names_to_disable.append(m["strategy"])
                audit_entries.append({
                    "strategy": m["strategy"],
                    "reason": "sharpe_below_zero_30d",
                    "sharpe": m["sharpe"],
                    "num_trades": m["num_trades"],
                    "total_pnl": m["total_pnl"],
                    "disabled_at": datetime.now(timezone.utc).isoformat(),
                })
            await session.commit()

        if names_to_disable:
            # Audit trail: structured log line per action + durable AgentMemory record.
            # (The relational audit_logs table requires a user FK; system actions
            # are recorded here instead until a nullable-actor migration exists.)
            for entry in audit_entries:
                logger.info(
                    "SelfImprovingLoop AUDIT: auto-disabled strategy=%s sharpe=%s num_trades=%s reason=%s",
                    entry["strategy"], entry["sharpe"], entry["num_trades"], entry["reason"],
                )
            await self._memory.write("auto_disabled", {
                "strategies": names_to_disable,
                "audit": audit_entries,
            })

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
            await self._memory.write("llm_suggestions", {
                "provider": response.provider,
                "suggestion": response.content,
            })
            logger.info("SelfImprovingLoop: LLM suggestion from %s stored", response.provider)

    # ── Regime broadcast ──────────────────────────────────────────────────────

    async def _broadcast_regime(self, metrics: list[dict]) -> None:
        profitable = sum(1 for m in metrics if m["sharpe"] > 0.5)
        total = len(metrics) or 1
        health = profitable / total

        regime = "bull" if health > 0.6 else ("bear" if health < 0.3 else "sideways")
        await self._memory.set_latest("platform_health", {
            "regime": regime,
            "health_ratio": round(health, 3),
            "profitable_strategies": profitable,
            "total_strategies": total,
        })

        try:
            await self._redis.publish("platform:regime", json.dumps({"regime": regime, "health": health}))
        except Exception:
            pass
