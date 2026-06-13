"""
Token budget manager for multi-agent collaboration.

Each agent has a daily token quota. Overspending is throttled and logged.
Budgets are tracked in Redis with daily TTL (resets at 00:00 UTC).

Usage:
    budget = TokenBudgetManager(redis_client)
    async with budget.reserve("strategy_agent", tokens=1000):
        # do LLM call
"""
from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from typing import AsyncIterator
from contextlib import asynccontextmanager

from app.utils.logging import logger


# Daily token quotas per agent (tune these to control API spend)
DEFAULT_BUDGETS: dict[str, int] = {
    "strategy_agent":    200_000,
    "ml_agent":          300_000,
    "risk_agent":         50_000,
    "research_agent":    400_000,
    "alpha_miner":       150_000,
    "backtest_agent":     50_000,
    "ui_agent":           50_000,
    "coordinator":       100_000,
    "default":           100_000,
}

# When remaining < WARN_THRESHOLD fraction, log a warning
WARN_THRESHOLD = 0.20


class TokenBudgetManager:
    def __init__(self, redis_client=None):
        self._redis = redis_client

    def _today_key(self, agent: str) -> str:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"token_budget:{agent}:{day}"

    async def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            from app.redis_client import get_redis
            return get_redis()
        except Exception:
            return None

    async def get_usage(self, agent: str) -> dict:
        """Returns {used, quota, remaining, pct_used}."""
        quota = DEFAULT_BUDGETS.get(agent, DEFAULT_BUDGETS["default"])
        redis = await self._get_redis()
        used = 0
        if redis:
            try:
                val = await redis.get(self._today_key(agent))
                used = int(val) if val else 0
            except Exception:
                pass
        remaining = max(0, quota - used)
        return {
            "agent": agent,
            "used": used,
            "quota": quota,
            "remaining": remaining,
            "pct_used": round(used / quota * 100, 1) if quota > 0 else 0.0,
        }

    async def can_spend(self, agent: str, tokens: int) -> bool:
        usage = await self.get_usage(agent)
        return usage["remaining"] >= tokens

    async def record_spend(self, agent: str, tokens: int) -> dict:
        """Record token spend. Returns updated usage dict."""
        quota = DEFAULT_BUDGETS.get(agent, DEFAULT_BUDGETS["default"])
        redis = await self._get_redis()
        used = tokens
        if redis:
            try:
                key = self._today_key(agent)
                used = await redis.incrby(key, tokens)
                # TTL: 25 hours so the key expires after the day rolls
                await redis.expire(key, 90_000)
            except Exception as e:
                logger.warning("TokenBudget: redis write failed", error=str(e))

        remaining = max(0, quota - used)
        if remaining < quota * WARN_THRESHOLD:
            logger.warning(
                "TokenBudget: approaching daily limit",
                agent=agent,
                used=used,
                quota=quota,
                remaining=remaining,
            )
        return {"agent": agent, "used": used, "quota": quota, "remaining": remaining}

    @asynccontextmanager
    async def reserve(self, agent: str, tokens: int) -> AsyncIterator[None]:
        """
        Async context manager: checks budget, yields, then records spend.
        Raises RuntimeError if agent is over budget.
        """
        if not await self.can_spend(agent, tokens):
            usage = await self.get_usage(agent)
            raise RuntimeError(
                f"Agent '{agent}' over daily token budget "
                f"({usage['used']}/{usage['quota']} used). "
                "Try again tomorrow or request a quota increase."
            )
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            await self.record_spend(agent, tokens)
            logger.debug(
                "TokenBudget: spend recorded",
                agent=agent,
                tokens=tokens,
                elapsed_ms=elapsed_ms,
            )

    async def all_usage(self) -> list[dict]:
        """Returns usage summary for all known agents."""
        return [await self.get_usage(a) for a in DEFAULT_BUDGETS if a != "default"]

    async def reset_agent(self, agent: str) -> None:
        """Manually reset an agent's daily budget (admin use)."""
        redis = await self._get_redis()
        if redis:
            try:
                await redis.delete(self._today_key(agent))
            except Exception:
                pass


_budget_manager: TokenBudgetManager | None = None


def get_token_budget() -> TokenBudgetManager:
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = TokenBudgetManager()
    return _budget_manager
