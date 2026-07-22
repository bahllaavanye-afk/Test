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
from typing import Any

logger = logging.getLogger(__name__)

_PREFIX = "agent:memory:"
_MAX_LIST_LEN = 500  # cap per topic to avoid unbounded growth


class AgentMemory:
    def __init__(self, redis_client: Any):
        self._r = redis_client

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, topic: str, data: dict) -> None:
        """Append an observation to a topic list with a timestamp."""
        start = time.perf_counter()
        payload = json.dumps({"ts": time.time(), **data})
        key = f"{_PREFIX}{topic}"
        try:
            await self._r.lpush(key, payload)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
            duration_ms = (time.perf_counter() - start) * 1000
            pnl = data.get("pnl")
            logger.info(
                "AgentMemory.write completed topic=%s count=1 duration_ms=%.2f pnl=%s",
                topic,
                duration_ms,
                pnl,
            )
        except Exception as e:
            logger.warning("AgentMemory.write failed for topic %s: %s", topic, e)

    async def set_latest(self, topic: str, data: dict) -> None:
        """Overwrite the latest value for a topic (single-value slot)."""
        start = time.perf_counter()
        key = f"{_PREFIX}latest:{topic}"
        payload = json.dumps({"ts": time.time(), **data})
        try:
            await self._r.set(key, payload)
            duration_ms = (time.perf_counter() - start) * 1000
            pnl = data.get("pnl")
            logger.info(
                "AgentMemory.set_latest completed topic=%s duration_ms=%.2f pnl=%s",
                topic,
                duration_ms,
                pnl,
            )
        except Exception as e:
            logger.warning("AgentMemory.set_latest failed for topic %s: %s", topic, e)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str, n: int = 50) -> list[dict]:
        """Return up to n most-recent observations for a topic."""
        start = time.perf_counter()
        key = f"{_PREFIX}{topic}"
        try:
            items = await self._r.lrange(key, 0, n - 1)
            observations = [json.loads(i) for i in items]
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "AgentMemory.read_recent completed topic=%s count=%d duration_ms=%.2f",
                topic,
                len(observations),
                duration_ms,
            )
            return observations
        except Exception as e:
            logger.warning("AgentMemory.read_recent failed for topic %s: %s", topic, e)
            return []

    async def get_latest(self, topic: str) -> dict | None:
        """Return the latest single-value for a topic."""
        start = time.perf_counter()
        key = f"{_PREFIX}latest:{topic}"
        try:
            val = await self._r.get(key)
            result = json.loads(val) if val else None
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "AgentMemory.get_latest completed topic=%s found=%s duration_ms=%.2f",
                topic,
                result is not None,
                duration_ms,
            )
            return result
        except Exception as e:
            logger.warning("AgentMemory.get_latest failed for topic %s: %s", topic, e)
            return None

    async def read_all_topics(self) -> list[str]:
        """List all memory topics currently stored."""
        start = time.perf_counter()
        try:
            pattern = f"{_PREFIX}*"
            keys = await self._r.keys(pattern)
            topics = [k.removeprefix(_PREFIX) for k in keys]
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "AgentMemory.read_all_topics completed count=%d duration_ms=%.2f",
                len(topics),
                duration_ms,
            )
            return topics
        except Exception as e:
            logger.warning("AgentMemory.read_all_topics failed: %s", e)
            return []