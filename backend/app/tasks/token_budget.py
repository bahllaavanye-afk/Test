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

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Dict

from pydantic import BaseModel, Field, PositiveInt, validator

from app.utils.logging import logger

# Daily token quotas per agent (tune these to control API spend)
DEFAULT_BUDGETS: dict[str, int] = {
    "strategy_agent": 200_000,
    "ml_agent": 300_000,
    "risk_agent": 50_000,
    "research_agent": 400_000,
    "alpha_miner": 150_000,
    "backtest_agent": 50_000,
    "ui_agent": 50_000,
    "coordinator": 100_000,
    "default": 100_000,
}

# When remaining < WARN_THRESHOLD fraction, log a warning
WARN_THRESHOLD = 0.20


class TokenUsage(BaseModel):
    """Schema representing token usage for a single agent."""

    agent: str = Field(
        ...,
        description="Identifier of the agent whose token usage is reported.",
        example="strategy_agent",
    )
    used: PositiveInt = Field(
        ...,
        description="Number of tokens already consumed today.",
        example=12345,
    )
    quota: PositiveInt = Field(
        ...,
        description="Daily token quota allocated to the agent.",
        example=200000,
    )
    remaining: int = Field(
        ...,
        description="Tokens remaining for the day (quota - used, never negative).",
        example=187655,
    )
    pct_used: float = Field(
        ...,
        description="Percentage of the quota that has been used, rounded to one decimal place.",
        example=6.2,
    )

    @validator("remaining")
    def non_negative_remaining(cls, v: int) -> int:
        if v < 0:
            raise ValueError("remaining must be non‑negative")
        return v

    @validator("pct_used")
    def pct_range(cls, v: float) -> float:
        if not (0.0 <= v <= 100.0):
            raise ValueError("pct_used must be between 0 and 100")
        return v

    class Config:
        schema_extra = {
            "example": {
                "agent": "strategy_agent",
                "used": 12345,
                "quota": 200000,
                "remaining": 187655,
                "pct_used": 6.2,
            }
        }


class TokenSpendRecord(BaseModel):
    """Schema returned after recording a token spend."""

    agent: str = Field(
        ...,
        description="Identifier of the agent for which the spend was recorded.",
        example="strategy_agent",
    )
    used: PositiveInt = Field(
        ...,
        description="Total tokens used after the spend.",
        example=12445,
    )
    quota: PositiveInt = Field(
        ...,
        description="Daily token quota allocated to the agent.",
        example=200000,
    )
    remaining: int = Field(
        ...,
        description="Tokens remaining after the spend.",
        example=187555,
    )

    class Config:
        schema_extra = {
            "example": {
                "agent": "strategy_agent",
                "used": 12445,
                "quota": 200000,
                "remaining": 187555,
            }
        }


class TokenBudgetManager:
    def __init__(self, redis_client=None):
        self._redis = redis_client

    def _today_key(self, agent: str) -> str:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        return f"token_budget:{agent}:{day}"

    async def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            from app.redis_client import get_redis

            return get_redis()
        except Exception:
            return None

    async def get_usage(self, agent: str) -> Dict[str, Any]:
        """Returns a validated dict with {agent, used, quota, remaining, pct_used}."""
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
        raw = {
            "agent": agent,
            "used": used,
            "quota": quota,
            "remaining": remaining,
            "pct_used": round(used / quota * 100, 1) if quota > 0 else 0.0,
        }
        # Validate and return plain dict to keep backward compatibility
        return TokenUsage(**raw).dict()

    async def can_spend(self, agent: str, tokens: int) -> bool:
        usage = await self.get_usage(agent)
        return usage["remaining"] >= tokens

    async def record_spend(self, agent: str, tokens: int) -> Dict[str, Any]:
        """Record token spend. Returns a validated usage dict."""
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
        raw = {"agent": agent, "used": used, "quota": quota, "remaining": remaining}
        return TokenSpendRecord(**raw).dict()

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

    async def all_usage(self) -> list[Dict[str, Any]]:
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