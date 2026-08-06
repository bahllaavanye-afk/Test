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
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_PREFIX = "agent:memory:"
_MAX_LIST_LEN = 500  # cap per topic to avoid unbounded growth


class AgentMemory:
    def __init__(self, redis_client: Any):
        self._r = redis_client

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, topic: str | None, data: Mapping[str, Any] | None) -> None:
        """Append an observation to a topic list with a timestamp."""
        if not topic:
            logger.warning("AgentMemory.write called with empty topic")
            return
        if not data:
            logger.warning("AgentMemory.write called with empty or None data for topic %s", topic)
            return
        try:
            payload = json.dumps({"ts": time.time(), **data})
            key = f"{_PREFIX}{topic}"
            await self._r.lpush(key, payload)
            # Trim to keep at most _MAX_LIST_LEN items (0-indexed inclusive)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
        except Exception as e:
            logger.warning("AgentMemory.write failed for topic %s: %s", topic, e)

    async def set_latest(self, topic: str | None, data: Mapping[str, Any] | None) -> None:
        """Overwrite the latest value for a topic (single-value slot)."""
        if not topic:
            logger.warning("AgentMemory.set_latest called with empty topic")
            return
        if not data:
            logger.warning("AgentMemory.set_latest called with empty or None data for topic %s", topic)
            return
        try:
            key = f"{_PREFIX}latest:{topic}"
            payload = json.dumps({"ts": time.time(), **data})
            await self._r.set(key, payload)
        except Exception as e:
            logger.warning("AgentMemory.set_latest failed for topic %s: %s", topic, e)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str | None, n: int = 50) -> list[dict]:
        """Return up to n most-recent observations for a topic."""
        if not topic:
            logger.warning("AgentMemory.read_recent called with empty topic")
            return []
        if not isinstance(n, int) or n <= 0:
            logger.warning("AgentMemory.read_recent called with non-positive n=%s for topic %s", n, topic)
            return []
        try:
            key = f"{_PREFIX}{topic}"
            # Redis lrange is inclusive; use n-1 to get at most n items
            items = await self._r.lrange(key, 0, n - 1)
            return [json.loads(i) for i in items]
        except Exception as e:
            logger.warning("AgentMemory.read_recent failed for topic %s: %s", topic, e)
            return []

    async def get_latest(self, topic: str | None) -> dict | None:
        """Return the latest single-value for a topic."""
        if not topic:
            logger.warning("AgentMemory.get_latest called with empty topic")
            return None
        try:
            key = f"{_PREFIX}latest:{topic}"
            val = await self._r.get(key)
            return json.loads(val) if val else None
        except Exception as e:
            logger.warning("AgentMemory.get_latest failed for topic %s: %s", topic, e)
            return None

    async def read_all_topics(self) -> list[str]:
        """List all memory topics currently stored."""
        try:
            pattern = f"{_PREFIX}*"
            keys = await self._r.keys(pattern)
            # Guard against None or empty result from Redis
            if not keys:
                return []
            return [k.removeprefix(_PREFIX) for k in keys]
        except Exception as e:
            logger.warning("AgentMemory.read_all_topics failed: %s", e)
            return []