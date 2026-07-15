"""
Redis-backed shared agent memory.

Agents read and write structured observations to a shared Redis namespace.
All data is JSON‑serialised. Keys are namespaced under ``agent:memory:``.

Usage:
    mem = AgentMemory(redis_client)
    await mem.write(
        "strategy_insight",
        {"strategy": "ema_stack_tv", "sharpe": 1.8}
    )
    observations = await mem.read_recent("strategy_insight", n=20)
    await mem.write(
        "market_regime",
        {"regime": "bull", "confidence": 0.85}
    )
    regime = await mem.get_latest("market_regime")
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, root_validator, validator

logger = logging.getLogger(__name__)

_PREFIX = "agent:memory:"
_MAX_LIST_LEN = 500  # cap per topic to avoid unbounded growth


class Observation(BaseModel):
    """
    Schema for a generic observation stored in Redis.

    Attributes
    ----------
    ts: float
        Unix timestamp when the observation was recorded.
    data: dict
        Arbitrary payload supplied by the caller.
    """

    ts: float = Field(
        ...,
        description="Unix timestamp of the observation (seconds since epoch).",
        example=1721234567.123,
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Arbitrary payload containing the observation details.",
        example={"strategy": "ema_stack_tv", "sharpe": 1.8},
    )

    @validator("ts")
    def ts_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timestamp must be a positive number")
        return v

    @root_validator
    def data_must_not_be_empty(cls, values):
        data = values.get("data")
        if not isinstance(data, dict) or not data:
            raise ValueError("data field must be a non‑empty dict")
        return values


class AgentMemory:
    def __init__(self, redis_client: Any):
        self._r = redis_client

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _prepare_payload(data: Union[Dict[str, Any], Observation]) -> str:
        """
        Validate ``data`` against the Observation schema and return a JSON string.

        Parameters
        ----------
        data: dict | Observation
            Payload to store. If a plain dict is provided it will be wrapped with
            the current timestamp and validated.

        Returns
        -------
        str
            JSON‑encoded payload ready for Redis storage.
        """
        if isinstance(data, Observation):
            payload_dict = data.dict()
        else:
            payload_dict = {"ts": time.time(), **data}
            # Validate using the Pydantic model
            Observation(**payload_dict)
        return json.dumps(payload_dict)

    # ── Write ────────────────────────────────────────────────────────────────

    async def write(self, topic: str, data: Union[Dict[str, Any], Observation]) -> None:
        """
        Append an observation to a topic list with a timestamp.

        Parameters
        ----------
        topic: str
            Topic name under ``agent:memory:``.
        data: dict | Observation
            Observation payload. If a dict is supplied it will be wrapped with a
            timestamp and validated against :class:`Observation`.
        """
        payload = self._prepare_payload(data)
        key = f"{_PREFIX}{topic}"
        try:
            await self._r.lpush(key, payload)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
        except Exception as e:  # pragma: no cover
            logger.warning("AgentMemory.write failed for topic %s: %s", topic, e)

    async def set_latest(self, topic: str, data: Union[Dict[str, Any], Observation]) -> None:
        """
        Overwrite the latest value for a topic (single‑value slot).

        Parameters
        ----------
        topic: str
            Topic name under ``agent:memory:latest:``.
        data: dict | Observation
            Observation payload. Handled the same way as :meth:`write`.
        """
        key = f"{_PREFIX}latest:{topic}"
        payload = self._prepare_payload(data)
        try:
            await self._r.set(key, payload)
        except Exception as e:  # pragma: no cover
            logger.warning("AgentMemory.set_latest failed for topic %s: %s", topic, e)

    # ── Read ───────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str, n: int = 50) -> List[Dict[str, Any]]:
        """
        Return up to ``n`` most‑recent observations for a topic.

        Parameters
        ----------
        topic: str
            Topic name.
        n: int, default 50
            Maximum number of items to retrieve.

        Returns
        -------
        list[dict]
            List of decoded observation dictionaries. Empty list on error.
        """
        key = f"{_PREFIX}{topic}"
        try:
            items = await self._r.lrange(key, 0, n - 1)
            return [json.loads(i) for i in items]
        except Exception as e:  # pragma: no cover
            logger.warning("AgentMemory.read_recent failed for topic %s: %s", topic, e)
            return []

    async def get_latest(self, topic: str) -> Union[Dict[str, Any], None]:
        """
        Return the latest single‑value for a topic.

        Parameters
        ----------
        topic: str
            Topic name.

        Returns
        -------
        dict | None
            Decoded observation payload or ``None`` if missing / on error.
        """
        key = f"{_PREFIX}latest:{topic}"
        try:
            val = await self._r.get(key)
            return json.loads(val) if val else None
        except Exception as e:  # pragma: no cover
            logger.warning("AgentMemory.get_latest failed for topic %s: %s", topic, e)
            return None

    async def read_all_topics(self) -> List[str]:
        """
        List all memory topics currently stored.

        Returns
        -------
        list[str]
            Topic names without the ``agent:memory:`` prefix.
        """
        try:
            pattern = f"{_PREFIX}*"
            keys = await self._r.keys(pattern)
            return [k.removeprefix(_PREFIX) for k in keys]
        except Exception as e:  # pragma: no cover
            logger.warning("AgentMemory.read_all_topics failed: %s", e)
            return []