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
import asyncio
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_PREFIX = "agent:memory:"
_MAX_LIST_LEN = 500  # cap per topic to avoid unbounded growth
_TOPICS_SET_KEY = f"{_PREFIX}topics"
_CACHE_TTL = 60  # seconds


class AgentMemory:
    def __init__(self, redis_client: Any):
        self._r = redis_client
        self._topics_cache: List[str] | None = None
        self._cache_timestamp: float = 0.0
        self._lock = asyncio.Lock()

    # ── Helper methods ────────────────────────────────────────────────────────

    def _key(self, topic: str, *, latest: bool = False) -> str:
        """Construct a Redis key for a given topic."""
        if not topic:
            raise ValueError("Topic must be a non‑empty string")
        prefix = f"{_PREFIX}latest:" if latest else _PREFIX
        return f"{prefix}{topic}"

    def _payload(self, data: dict) -> str:
        """Serialise data with a timestamp."""
        if data is None:
            data = {}
        return json.dumps({"ts": time.time(), **data})

    async def _log_error(self, operation: str, topic: str | None, exc: Exception) -> None:
        """Log a Redis operation failure."""
        if topic:
            logger.warning("AgentMemory.%s failed for topic %s: %s", operation, topic, exc)
        else:
            logger.warning("AgentMemory.%s failed: %s", operation, exc)

    def _log_info(
        self,
        operation: str,
        topic: str,
        count: int,
        duration: float,
        pnl: Optional[Any] = None,
    ) -> None:
        """Log successful operation metrics at INFO level."""
        msg = (
            f"AgentMemory.{operation} topic={topic} count={count} "
            f"duration={duration:.6f}s"
        )
        if pnl is not None:
            msg += f" pnl={pnl}"
        logger.info(msg)

    async def _add_topic_to_set(self, topic: str) -> None:
        """Ensure the topic is recorded in the Redis set of topics."""
        if not topic:
            await self._log_error("add_topic_to_set", topic, ValueError("Invalid topic"))
            return
        try:
            await self._r.sadd(_TOPICS_SET_KEY, topic)
        except Exception as e:
            await self._log_error("add_topic_to_set", topic, e)

    def _decode_topics(self, raw_topics: Any) -> List[str]:
        """Convert raw Redis topic members to a list of strings."""
        return [
            t.decode() if isinstance(t, (bytes, bytearray)) else str(t)
            for t in raw_topics
        ]

    def _update_topics_cache(self, topics: List[str]) -> None:
        """Refresh the in‑memory cache of topics."""
        self._topics_cache = topics
        self._cache_timestamp = time.time()

    async def _fetch_topics_from_redis(self) -> List[str]:
        """Retrieve the set of topics from Redis, handling empty results."""
        raw_topics = await self._r.smembers(_TOPICS_SET_KEY)
        if not raw_topics:
            return []
        return self._decode_topics(raw_topics)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, topic: str, data: dict) -> None:
        """Append an observation to a topic list with a timestamp."""
        if not topic:
            await self._log_error("write", topic, ValueError("Topic cannot be empty"))
            return
        payload = self._payload(data)
        key = self._key(topic)
        start = time.time()
        try:
            await self._r.lpush(key, payload)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
            await self._add_topic_to_set(topic)
            duration = time.time() - start
            pnl = data.get("pnl")
            self._log_info("write", topic, count=1, duration=duration, pnl=pnl)
        except Exception as e:
            await self._log_error("write", topic, e)

    async def set_latest(self, topic: str, data: dict) -> None:
        """Overwrite the latest value for a topic (single-value slot)."""
        if not topic:
            await self._log_error("set_latest", topic, ValueError("Topic cannot be empty"))
            return
        payload = self._payload(data)
        key = self._key(topic, latest=True)
        start = time.time()
        try:
            await self._r.set(key, payload)
            await self._add_topic_to_set(topic)
            duration = time.time() - start
            pnl = data.get("pnl")
            self._log_info("set_latest", topic, count=1, duration=duration, pnl=pnl)
        except Exception as e:
            await self._log_error("set_latest", topic, e)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str, n: int = 50) -> List[dict]:
        """Return up to n most‑recent observations for a topic."""
        if not topic:
            await self._log_error("read_recent", topic, ValueError("Topic cannot be empty"))
            return []
        if n <= 0:
            return []
        key = self._key(topic)
        start = time.time()
        try:
            items = await self._r.lrange(key, 0, n - 1)
            result = [json.loads(i) for i in items]
            duration = time.time() - start
            pnl_vals = [item.get("pnl") for item in result if "pnl" in item]
            pnl = sum(pnl_vals) if pnl_vals else None
            self._log_info(
                "read_recent", topic, count=len(result), duration=duration, pnl=pnl
            )
            return result
        except Exception as e:
            await self._log_error("read_recent", topic, e)
            return []

    async def get_latest(self, topic: str) -> dict | None:
        """Return the latest single‑value for a topic."""
        if not topic:
            await self._log_error("get_latest", topic, ValueError("Topic cannot be empty"))
            return None
        key = self._key(topic, latest=True)
        start = time.time()
        try:
            val = await self._r.get(key)
            result = json.loads(val) if val else None
            duration = time.time() - start
            pnl = result.get("pnl") if isinstance(result, dict) else None
            count = 1 if result else 0
            self._log_info(
                "get_latest", topic, count=count, duration=duration, pnl=pnl
            )
            return result
        except Exception as e:
            await self._log_error("get_latest", topic, e)
            return None

    async def read_all_topics(self) -> List[str]:
        """List all memory topics currently stored, using a short‑lived cache."""
        async with self._lock:
            now = time.time()
            if self._topics_cache is not None and (now - self._cache_timestamp) < _CACHE_TTL:
                return self._topics_cache

            try:
                topics = await self._fetch_topics_from_redis()
                self._update_topics_cache(topics)
                return topics
            except Exception as e:
                await self._log_error("read_all_topics", None, e)
                return []