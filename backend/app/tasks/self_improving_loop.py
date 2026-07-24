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

    # ── Auto-disable ──────────────────────────────────────────────────────────

    async def _auto_disable_underperformers(self, metrics: list[dict]) -> None:
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
            logger.info(
                "SelfImprovingLoop: LLM suggestion from %s stored", response.provider
            )

    # ── Regime broadcast ──────────────────────────────────────────────────────

    async def _broadcast_regime(self, metrics: list[dict]) -> None:
        profitable = sum(1 for m in metrics if m["sharpe"] > 0.5)
        total = len(metrics) or 1
        health = profitable / total

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
                "platform:regime", json.dumps({"regime": regime, "health": health})
            )
        except Exception as exc:  # noqa: BLE001 — subscribers just miss one regime tick
            logger.debug("self-improving loop: regime publish failed: %s", exc)


# --------------------------------------------------------------------------- #
# Unit tests for edge‑case behavior
# --------------------------------------------------------------------------- #

import pytest
from unittest.mock import AsyncMock, MagicMock


class _FakeRow:
    """Simple attribute container to mimic SQLAlchemy row objects."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeSession:
    """Async context manager that records executed statements."""

    def __init__(self, result: _FakeResult | None = None):
        self.result = result or _FakeResult([])
        self.executed = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, *args, **kwargs):
        self.executed.append((args, kwargs))
        return self.result

    async def commit(self):
        self.committed = True


@pytest.mark.asyncio
async def test_collect_strategy_metrics_returns_empty_when_no_data():
    """Boundary: no rows should produce an empty list without errors."""
    async_factory = MagicMock(return_value=_FakeSession())
    loop = SelfImprovingLoop(db_session_factory=async_factory, redis_client=AsyncMock())
    metrics = await loop._collect_strategy_metrics()
    assert metrics == []  # should be empty list


@pytest.mark.asyncio
async def test_auto_disable_underperformers_boundary_conditions():
    """Sharpe == 0 must NOT disable; Sharpe < 0 with exactly 10 trades must disable."""
    # Prepare metrics with three strategies:
    # 1. Negative Sharpe, exactly 10 trades -> should be disabled
    # 2. Negative Sharpe, fewer than 10 trades -> should NOT be disabled
    # 3. Sharpe == 0, 15 trades -> should NOT be disabled
    metrics = [
        {"strategy": "neg10", "sharpe": -0.01, "num_trades": 10},
        {"strategy": "neg9", "sharpe": -0.5, "num_trades": 9},
        {"strategy": "zero", "sharpe": 0.0, "num_trades": 15},
    ]

    fake_session = _FakeSession()
    async_factory = MagicMock(return_value=fake_session)
    redis_mock = AsyncMock()
    loop = SelfImprovingLoop(db_session_factory=async_factory, redis_client=redis_mock)

    # Patch memory to avoid external calls
    loop._memory = AsyncMock()

    await loop._auto_disable_underperformers(metrics)

    # Only one UPDATE should have been executed for "neg10"
    assert len(fake_session.executed) == 1
    executed_sql, params = fake_session.executed[0]
    assert "UPDATE strategies" in executed_sql[0]
    assert params["name"] == "neg10"
    assert fake_session.committed is True

    # Verify memory write recorded the correct strategy name
    loop._memory.write.assert_awaited_once_with(
        "auto_disabled", {"strategies": ["neg10"]}
    )


@pytest.mark.asyncio
async def test_broadcast_regime_boundary_regime_selection():
    """Health exactly at thresholds (0.6, 0.3) should map to 'sideways'."""
    # Helper to capture the regime written to memory
    memory_mock = AsyncMock()
    redis_mock = AsyncMock()

    async_factory = MagicMock(return_value=_FakeSession())
    loop = SelfImprovingLoop(db_session_factory=async_factory, redis_client=redis_mock)
    loop._memory = memory_mock

    # Case 1: health = 0.6 (5 profitable out of 8)
    metrics = [
        {"strategy": f"s{i}", "sharpe": 0.7 if i < 5 else 0.4}
        for i in range(8)
    ]
    await loop._broadcast_regime(metrics)
    # Expect "sideways" because health is not > 0.6
    memory_mock.set_latest.assert_awaited_once()
    args, kwargs = memory_mock.set_latest.call_args
    assert args[0] == "platform_health"
    assert args[1]["regime"] == "sideways"

    # Reset mock for next case
    memory_mock.reset_mock()

    # Case 2: health = 0.3 (3 profitable out of 10)
    metrics = [
        {"strategy": f"s{i}", "sharpe": 0.8 if i < 3 else 0.2}
        for i in range(10)
    ]
    await loop._broadcast_regime(metrics)
    args, kwargs = memory_mock.set_latest.call_args
    assert args[0] == "platform_health"
    assert args[1]["regime"] == "sideways"