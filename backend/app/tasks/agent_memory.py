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
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_PREFIX = "agent:memory:"
_MAX_LIST_LEN = 500  # cap per topic to avoid unbounded growth


def _decode_if_needed(value: Any) -> Any:
    """Decode bytes to str for JSON handling; pass through otherwise."""
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode()
        except Exception as e:
            logger.debug("Failed to decode Redis bytes: %s", e)
            return value
    return value


class AgentMemory:
    def __init__(self, redis_client: Any):
        self._r = redis_client

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, topic: Optional[str], data: Optional[dict]) -> None:
        """Append an observation to a topic list with a timestamp.

        Safely handles None or empty inputs; logs and returns early.
        """
        if not topic:
            logger.warning("AgentMemory.write called with empty topic.")
            return
        if not data:
            logger.warning("AgentMemory.write called with empty data for topic %s.", topic)
            return

        payload = json.dumps({"ts": time.time(), **data})
        key = f"{_PREFIX}{topic}"
        try:
            await self._r.lpush(key, payload)
            # ltrim keeps indices 0 through _MAX_LIST_LEN-1 inclusive
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
        except Exception as e:
            logger.warning("AgentMemory.write failed for topic %s: %s", topic, e)

    async def set_latest(self, topic: Optional[str], data: Optional[dict]) -> None:
        """Overwrite the latest value for a topic (single-value slot).

        Handles None or empty inputs gracefully.
        """
        if not topic:
            logger.warning("AgentMemory.set_latest called with empty topic.")
            return
        if not data:
            logger.warning(
                "AgentMemory.set_latest called with empty data for topic %s.", topic
            )
            return

        key = f"{_PREFIX}latest:{topic}"
        payload = json.dumps({"ts": time.time(), **data})
        try:
            await self._r.set(key, payload)
        except Exception as e:
            logger.warning("AgentMemory.set_latest failed for topic %s: %s", topic, e)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: Optional[str], n: int = 50) -> List[dict]:
        """Return up to n most-recent observations for a topic.

        Handles None/empty topic and non‑positive n values.
        """
        if not topic:
            logger.warning("AgentMemory.read_recent called with empty topic.")
            return []
        if n <= 0:
            return []

        key = f"{_PREFIX}{topic}"
        try:
            items = await self._r.lrange(key, 0, n - 1)
            return [json.loads(_decode_if_needed(i)) for i in items]
        except Exception as e:
            logger.warning("AgentMemory.read_recent failed for topic %s: %s", topic, e)
            return []

    async def get_latest(self, topic: Optional[str]) -> Optional[dict]:
        """Return the latest single-value for a topic."""
        if not topic:
            logger.warning("AgentMemory.get_latest called with empty topic.")
            return None

        key = f"{_PREFIX}latest:{topic}"
        try:
            val = await self._r.get(key)
            return json.loads(_decode_if_needed(val)) if val else None
        except Exception as e:
            logger.warning("AgentMemory.get_latest failed for topic %s: %s", topic, e)
            return None

    async def read_all_topics(self) -> List[str]:
        """List all memory topics currently stored."""
        try:
            pattern = f"{_PREFIX}*"
            keys = await self._r.keys(pattern)
            # Keys may be bytes; ensure they are strings before stripping prefix
            cleaned = [
                (k.decode() if isinstance(k, (bytes, bytearray)) else str(k)).removeprefix(
                    _PREFIX
                )
                for k in keys
            ]
            return cleaned
        except Exception as e:
            logger.warning("AgentMemory.read_all_topics failed: %s", e)
            return []