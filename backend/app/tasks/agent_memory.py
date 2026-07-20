"""
Redis-backed shared agent memory.

Agents read and write structured observations to a shared Redis namespace.
All data is JSON-serialised. Keys are namespaced under 'agent:memory:'.

Usage:
    mem = AgentMemory(redis_client)
    await mem.write("strategy_insight", StrategyInsight(strategy="ema_stack_tv", sharpe=1.8))
    observations = await mem.read_recent("strategy_insight", n=20)
    await mem.write("market_regime", MarketRegime(regime="bull", confidence=0.85))
    regime = await mem.get_latest("market_regime")
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, PositiveFloat, validator

logger = logging.getLogger(__name__)

_PREFIX = "agent:memory:"
_MAX_LIST_LEN = 500  # cap per topic to avoid unbounded growth


class ObservationBase(BaseModel):
    """Base model for any observation stored in AgentMemory.

    Attributes:
        ts: Unix timestamp (seconds since epoch) when the observation was created.
    """

    ts: PositiveFloat = Field(
        ...,
        description="Unix timestamp (seconds since epoch) when the observation was recorded.",
        examples=[1680307200.0],
    )

    @validator("ts")
    def _ensure_non_future(cls, v: float) -> float:
        """Ensure the timestamp is not absurdly far in the future."""
        now = time.time()
        if v > now + 60 * 60:  # allow a 1‑hour clock skew
            raise ValueError("timestamp is too far in the future")
        return v

    class Config:
        extra = "allow"


class StrategyInsight(ObservationBase):
    """Observation describing a trading strategy insight.

    Attributes:
        strategy: Identifier of the strategy.
        sharpe: Estimated Sharpe ratio of the strategy.
    """

    strategy: str = Field(
        ...,
        description="Identifier of the trading strategy.",
        examples=["ema_stack_tv"],
    )
    sharpe: PositiveFloat = Field(
        ...,
        description="Estimated Sharpe ratio of the strategy.",
        examples=[1.8],
    )

    @validator("strategy")
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("strategy must be a non‑empty string")
        return v


class MarketRegime(ObservationBase):
    """Observation describing the current market regime.

    Attributes:
        regime: Market regime classification (e.g., bull, bear, sideways).
        confidence: Confidence level of the classification, between 0 and 1.
    """

    regime: str = Field(
        ...,
        description="Market regime classification (e.g., bull, bear, sideways).",
        examples=["bull"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of the regime classification, inclusive of 0 and 1.",
        examples=[0.85],
    )

    @validator("regime")
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("regime must be a non‑empty string")
        return v


class AgentMemory:
    def __init__(self, redis_client: Any):
        self._r = redis_client

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, topic: str, data: Union[Dict[str, Any], BaseModel]) -> None:
        """Append an observation to a topic list with a timestamp.

        Args:
            topic: The memory topic name.
            data: Observation data, either a raw dict or a Pydantic model.
        """
        payload_dict = data.dict() if isinstance(data, BaseModel) else data
        # Ensure a timestamp is present; if missing, add the current time.
        if "ts" not in payload_dict:
            payload_dict["ts"] = time.time()
        payload = json.dumps(payload_dict)
        key = f"{_PREFIX}{topic}"
        try:
            await self._r.lpush(key, payload)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
        except Exception as e:
            logger.warning("AgentMemory.write failed for topic %s: %s", topic, e)

    async def set_latest(self, topic: str, data: Union[Dict[str, Any], BaseModel]) -> None:
        """Overwrite the latest value for a topic (single-value slot).

        Args:
            topic: The memory topic name.
            data: Observation data, either a raw dict or a Pydantic model.
        """
        key = f"{_PREFIX}latest:{topic}"
        payload_dict = data.dict() if isinstance(data, BaseModel) else data
        if "ts" not in payload_dict:
            payload_dict["ts"] = time.time()
        payload = json.dumps(payload_dict)
        try:
            await self._r.set(key, payload)
        except Exception as e:
            logger.warning("AgentMemory.set_latest failed for topic %s: %s", topic, e)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str, n: int = 50) -> List[Dict[str, Any]]:
        """Return up to n most-recent observations for a topic.

        Args:
            topic: The memory topic name.
            n: Maximum number of observations to retrieve.

        Returns:
            A list of observation dictionaries ordered from newest to oldest.
        """
        key = f"{_PREFIX}{topic}"
        try:
            items = await self._r.lrange(key, 0, n - 1)
            return [json.loads(i) for i in items]
        except Exception as e:
            logger.warning("AgentMemory.read_recent failed for topic %s: %s", topic, e)
            return []

    async def get_latest(self, topic: str) -> Union[Dict[str, Any], None]:
        """Return the latest single-value for a topic.

        Args:
            topic: The memory topic name.

        Returns:
            The most recent observation dictionary, or None if not present.
        """
        key = f"{_PREFIX}latest:{topic}"
        try:
            val = await self._r.get(key)
            return json.loads(val) if val else None
        except Exception as e:
            logger.warning("AgentMemory.get_latest failed for topic %s: %s", topic, e)
            return None

    async def read_all_topics(self) -> List[str]:
        """List all memory topics currently stored.

        Returns:
            A list of topic names without the prefix.
        """
        try:
            pattern = f"{_PREFIX}*"
            keys = await self._r.keys(pattern)
            return [k.removeprefix(_PREFIX) for k in keys]
        except Exception as e:
            logger.warning("AgentMemory.read_all_topics failed: %s", e)
            return []