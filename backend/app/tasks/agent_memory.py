"""
Redis-backed shared agent memory.

Agents read and write structured observations to a shared Redis namespace.
All data is JSON-serialised. Keys are namespaced under 'agent:memory:'.

Usage:
    mem = AgentMemory(redis_client)
    await mem.write("strategy_insight", {"strategy": "ema_stack_tv", "sharpe": 1.8})
    observations = await mem.read_recent("strategy_insight", n=20)
    await mem.write("market_regime", {"regime": "bull", "confidence": 0.85})
    regime = await mem.get_latest("market_regime")
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

_PREFIX = "agent:memory:"
_MAX_LIST_LEN = 500  # cap per topic to avoid unbounded growth


class BaseObservation(BaseModel):
    """Base model for all agent memory observations.

    Attributes
    ----------
    ts: float
        Unix timestamp of the observation.
    """

    ts: float = Field(
        ...,
        description="Unix timestamp of the observation",
        example=1627847265.123,
    )

    class Config:
        extra = "allow"
        schema_extra = {"example": {"ts": 1627847265.123, "key": "value"}}

    @validator("ts")
    def _validate_ts(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Timestamp must be positive")
        return v


class StrategyInsight(BaseObservation):
    """Observation for a strategy insight.

    Attributes
    ----------
    strategy: str
        Name of the strategy.
    sharpe: float
        Sharpe ratio of the strategy.
    """

    strategy: str = Field(
        ...,
        description="Name of the strategy",
        example="ema_stack_tv",
    )
    sharpe: float = Field(
        ...,
        description="Sharpe ratio of the strategy",
        example=1.8,
    )

    class Config:
        schema_extra = {
            "example": {
                "ts": 1627847265.123,
                "strategy": "ema_stack_tv",
                "sharpe": 1.8,
            }
        }


class MarketRegime(BaseObservation):
    """Observation describing the current market regime.

    Attributes
    ----------
    regime: str
        Market regime identifier.
    confidence: float
        Confidence level of the regime classification (0‑1).
    """

    regime: str = Field(
        ...,
        description="Market regime identifier",
        example="bull",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of the regime classification",
        example=0.85,
    )

    class Config:
        schema_extra = {
            "example": {
                "ts": 1627847265.123,
                "regime": "bull",
                "confidence": 0.85,
            }
        }


class AgentMemory:
    def __init__(self, redis_client: Any):
        self._r = redis_client

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, topic: str, data: Union[Dict[str, Any], BaseModel]) -> None:
        """Append an observation to a topic list with a timestamp.

        Parameters
        ----------
        topic: str
            The memory topic name.
        data: dict | BaseModel
            Observation data; a Pydantic model is accepted for validation.
        """
        payload_dict = (
            data.dict() if isinstance(data, BaseModel) else dict(data)
        )
        payload_dict["ts"] = time.time()
        # Validate using the generic BaseObservation to ensure timestamp correctness.
        BaseObservation(**payload_dict)

        payload = json.dumps(payload_dict)
        key = f"{_PREFIX}{topic}"
        try:
            await self._r.lpush(key, payload)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
        except Exception as e:
            logger.warning("AgentMemory.write failed for topic %s: %s", topic, e)

    async def set_latest(self, topic: str, data: Union[Dict[str, Any], BaseModel]) -> None:
        """Overwrite the latest value for a topic (single-value slot).

        Parameters
        ----------
        topic: str
            The memory topic name.
        data: dict | BaseModel
            Observation data; a Pydantic model is accepted for validation.
        """
        payload_dict = (
            data.dict() if isinstance(data, BaseModel) else dict(data)
        )
        payload_dict["ts"] = time.time()
        BaseObservation(**payload_dict)

        payload = json.dumps(payload_dict)
        key = f"{_PREFIX}latest:{topic}"
        try:
            await self._r.set(key, payload)
        except Exception as e:
            logger.warning("AgentMemory.set_latest failed for topic %s: %s", topic, e)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str, n: int = 50) -> List[Dict[str, Any]]:
        """Return up to n most-recent observations for a topic.

        Parameters
        ----------
        topic: str
            The memory topic name.
        n: int, optional
            Maximum number of observations to return (default 50).

        Returns
        -------
        list[dict]
            List of observation dictionaries ordered from newest to oldest.
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

        Parameters
        ----------
        topic: str
            The memory topic name.

        Returns
        -------
        dict | None
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

        Returns
        -------
        list[str]
            Topic names without the ``agent:memory:`` prefix.
        """
        try:
            pattern = f"{_PREFIX}*"
            keys = await self._r.keys(pattern)
            return [k.removeprefix(_PREFIX) for k in keys]
        except Exception as e:
            logger.warning("AgentMemory.read_all_topics failed: %s", e)
            return []