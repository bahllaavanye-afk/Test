"""
Redis-backed shared agent memory.

Agents read and write structured observations to a shared Redis namespace.
All data is JSON‑serialised. Keys are namespaced under ``agent:memory:``.

Typical usage::
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
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

_PREFIX = "agent:memory:"
_MAX_LIST_LEN = 500  # cap per topic to avoid unbounded growth


class AgentMemory:
    """Asynchronous interface for storing and retrieving agent observations in Redis.

    The memory stores two kinds of structures:

    * **Topic lists** – a capped list of timestamped observations per topic.
    * **Latest slot** – a single most‑recent value per topic, stored under a ``latest:`` prefix.
    """

    def __init__(self, redis_client: Any):
        """Create an ``AgentMemory`` instance.

        Args:
            redis_client: An asynchronous Redis client supporting ``lpush``, ``ltrim``,
                ``lrange``, ``set``, ``get``, ``keys`` and related methods.
        """
        self._r = redis_client

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, topic: str, data: Mapping[str, Any]) -> None:
        """Append an observation to a topic list.

        The observation is JSON‑serialised and prefixed with a ``ts`` field containing the
        current UNIX timestamp.

        Args:
            topic: The logical name of the observation stream.
            data:  Mapping of observation fields (excluding ``ts``).

        """
        payload = json.dumps({"ts": time.time(), **data})
        key = f"{_PREFIX}{topic}"
        try:
            await self._r.lpush(key, payload)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
        except Exception as e:
            logger.warning("AgentMemory.write failed for topic %s: %s", topic, e)

    async def set_latest(self, topic: str, data: Mapping[str, Any]) -> None:
        """Overwrite the latest value for a topic (single‑value slot).

        The value is JSON‑serialised and includes a ``ts`` timestamp.

        Args:
            topic: The logical name of the topic.
            data:  Mapping of fields to store (excluding ``ts``).
        """
        key = f"{_PREFIX}latest:{topic}"
        payload = json.dumps({"ts": time.time(), **data})
        try:
            await self._r.set(key, payload)
        except Exception as e:
            logger.warning("AgentMemory.set_latest failed for topic %s: %s", topic, e)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str, n: int = 50) -> List[Dict[str, Any]]:
        """Return up to ``n`` most‑recent observations for a topic.

        Args:
            topic: The logical name of the observation stream.
            n:     Maximum number of items to retrieve.

        Returns:
            A list of deserialised observations ordered from newest to oldest.
            Returns an empty list on error.
        """
        key = f"{_PREFIX}{topic}"
        try:
            items = await self._r.lrange(key, 0, n - 1)
            return [json.loads(i) for i in items]
        except Exception as e:
            logger.warning("AgentMemory.read_recent failed for topic %s: %s", topic, e)
            return []

    async def get_latest(self, topic: str) -> Optional[Dict[str, Any]]:
        """Return the latest single‑value for a topic.

        Args:
            topic: The logical name of the topic.

        Returns:
            The deserialised latest value, or ``None`` if the key does not exist or an error occurs.
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
            A list of topic names (without the ``agent:memory:`` prefix).
            Returns an empty list on error.
        """
        try:
            pattern = f"{_PREFIX}*"
            keys = await self._r.keys(pattern)
            return [k.removeprefix(_PREFIX) for k in keys]
        except Exception as e:
            logger.warning("AgentMemory.read_all_topics failed: %s", e)
            return []