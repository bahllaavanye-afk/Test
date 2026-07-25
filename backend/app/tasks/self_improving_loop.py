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
from typing import Any, List, Tuple

from pydantic import BaseModel, Field, validator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasks.free_llm_router import call_race
from app.tasks.agent_memory import AgentMemory

logger = logging.getLogger(__name__)


class StrategyMetric(BaseModel):
    """Metrics for a single strategy over the last 30 days."""

    strategy: str = Field(..., description="Name of the strategy.", example="mean_rev_20_2")
    num_trades: int = Field(
        ..., ge=0, description="Number of trades executed in the period.", example=42
    )
    total_pnl: float = Field(..., description="Total profit & loss.", example=1250.75)
    avg_pnl: float = Field(..., description="Average profit per trade.", example=29.78)
    win_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of winning trades (0‑1).",
        example=0.62,
    )
    sharpe: float = Field(..., description="Annualized Sharpe ratio.", example=1.23)

    @validator("sharpe", pre=True, always=True)
    def round_sharpe(cls, v: float) -> float:
        """Round Sharpe to three decimal places."""
        return round(v, 3)


class AutoDisabledPayload(BaseModel):
    """Payload describing strategies that were auto‑disabled."""

    strategies: List[str] = Field(
        ..., description="List of strategy names that were disabled.", example=["mean_rev_20_2"]
    )


class LLMSuggestion(BaseModel):
    """LLM suggestion payload stored in AgentMemory."""

    provider: str = Field(..., description="LLM provider identifier.", example="openai")
    suggestion: str = Field(..., description="Raw suggestion text from the LLM.", example="Adjust lookback period.")


class RegimePayload(BaseModel):
    """Broadcast payload describing current market regime."""

    regime: str = Field(
        ...,
        description="Current market regime classification.",
        example="bull",
        regex="^(bull|bear|sideways)$",
    )
    health_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proportion of strategies with Sharpe > 0.5.",
        example=0.71,
    )
    profitable_strategies: int = Field(..., ge=0, description="Number of profitable strategies.", example=14)
    total_strategies: int = Field(..., ge=1, description="Total number of evaluated strategies.", example=20)


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
            logger.info(
                "SelfImprovingLoop: cycle complete (%d strategies evaluated)", len(metrics)
            )
        except Exception as e:
            logger.exception("SelfImprovingLoop cycle error: %s", e)

    # ── Metric collection ─────────────────────────────────────────────────────

    async def _collect_strategy_metrics(self) -> List[dict]:
        """Pull per‑strategy Sharpe + win‑rate from trade history (last 30d)."""
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

        metrics: List[dict] = []
        for row in rows:
            std = row.std_pnl or 1e-9
            sharpe = (row.avg_pnl / std) * (252 ** 0.5) if std > 0 else 0
            metric = StrategyMetric(
                strategy=row.strategy_name,
                num_trades=row.num_trades,
                total_pnl=float(row.total_pnl or 0),
                avg_pnl=float(row.avg_pnl or 0),
                win_rate=float(row.win_rate or 0),
                sharpe=sharpe,
            )
            metrics.append(metric.dict())
        return metrics

    # ── Auto-disable ──────────────────────────────────────────────────────────

    async def _auto_disable_underperformers(self, metrics: List[dict]) -> None:
        """Disable strategies with Sharpe < 0 and >= 10 trades in the last 30 days."""
        underperformers = [
            m for m in metrics if m["sharpe"] < 0 and m["num_trades"] >= 10
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
        payload = AutoDisabledPayload(strategies=names)
        await self._memory.write("auto_disabled", payload.dict())

    # ── LLM improvement pass ──────────────────────────────────────────────────

    async def _llm_improvement_pass(self, metrics: List[dict]) -> None:
        if not metrics:
            return

        top, bottom = self._select_top_bottom_strategies(metrics)
        prompt = self._build_llm_prompt(top, bottom)

        response = await call_race(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=512,
        )
        if response:
            await self._store_llm_suggestion(response)

    def _select_top_bottom_strategies(self, metrics: List[dict]) -> Tuple[List[dict], List[dict]]:
        """Return the top 5 and bottom 3 strategies based on Sharpe."""
        top = sorted(metrics, key=lambda m: m["sharpe"], reverse=True)[:5]
        bottom = sorted(metrics, key=lambda m: m["sharpe"])[:3]
        return top, bottom

    def _build_llm_prompt(self, top: List[dict], bottom: List[dict]) -> str:
        """Create the prompt sent to the LLM with formatted strategy data."""
        return f"""You are a quantitative trading researcher.

Top performing strategies (last 30d):
{json.dumps(top, indent=2)}

Underperforming strategies:
{json.dumps(bottom, indent=2)}

Suggest 3 specific, actionable improvements:
1. Parameter tuning for the worst performer
2. A new indicator combination to test
3. A risk rule change to protect capital

Be concise. Each suggestion under 2 sentences."""

    async def _store_llm_suggestion(self, response: Any) -> None:
        """Persist LLM suggestion to AgentMemory and log the provider."""
        suggestion = LLMSuggestion(provider=response.provider, suggestion=response.content)
        await self._memory.write("llm_suggestions", suggestion.dict())
        logger.info("SelfImprovingLoop: LLM suggestion from %s stored", response.provider)

    # ── Regime broadcast ──────────────────────────────────────────────────────

    async def _broadcast_regime(self, metrics: List[dict]) -> None:
        profitable = sum(1 for m in metrics if m["sharpe"] > 0.5)
        total = len(metrics) or 1
        health = profitable / total

        regime = "bull" if health > 0.6 else ("bear" if health < 0.3 else "sideways")
        payload = RegimePayload(
            regime=regime,
            health_ratio=health,
            profitable_strategies=profitable,
            total_strategies=total,
        )
        await self._memory.set_latest("platform_health", payload.dict())

        try:
            await self._redis.publish(
                "platform:regime", json.dumps({"regime": regime, "health": health})
            )
        except Exception as exc:  # noqa: BLE001 — subscribers just miss one regime tick
            logger.debug("self-improving loop: regime publish failed: %s", exc)