"""
Redis‑backed shared agent memory.

Agents read and write structured observations to a shared Redis namespace.
All data is JSON‑serialised. Keys are namespaced under ``agent:memory:``.
The memory supports per‑topic list storage as well as a single‑latest slot.

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
import asyncio
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PREFIX = "agent:memory:"
_MAX_LIST_LEN = 500  # cap per topic to avoid unbounded growth
_TOPICS_SET_KEY = f"{_PREFIX}topics"
_CACHE_TTL = 60  # seconds


class AgentMemory:
    """Utility class for persisting and retrieving observations in Redis.

    The class maintains a small in‑memory cache of the known topics to avoid
    frequent Redis round‑trips. All write operations also ensure the topic is
    recorded in a Redis set so that :meth:`read_all_topics` can enumerate them.
    """

    def __init__(self, redis_client: Any):
        """Create an ``AgentMemory`` instance.

        Args:
            redis_client: An async Redis client supporting ``lpush``, ``ltrim``,
                ``set``, ``get``, ``lrange``, ``smembers`` and ``sadd``.
        """
        self._r = redis_client
        self._topics_cache: Optional[List[str]] = None
        self._cache_timestamp: float = 0.0
        self._lock = asyncio.Lock()

    # ── Helper methods ────────────────────────────────────────────────────────

    def _key(self, topic: str, *, latest: bool = False) -> str:
        """Construct a Redis key for *topic*.

        Args:
            topic: The memory topic name.
            latest: If ``True``, returns the key for the single‑latest slot;
                otherwise returns the list key.

        Returns:
            The fully‑qualified Redis key.

        Raises:
            ValueError: If *topic* is empty.
        """
        if not topic:
            raise ValueError("Topic must be a non‑empty string")
        prefix = f"{_PREFIX}latest:" if latest else _PREFIX
        return f"{prefix}{topic}"

    def _payload(self, data: Dict[str, Any]) -> str:
        """Serialise *data* together with a timestamp.

        Args:
            data: Arbitrary JSON‑serialisable payload.

        Returns:
            A JSON string containing the payload and a ``ts`` field.
        """
        if data is None:
            data = {}
        return json.dumps({"ts": time.time(), **data})

    async def _log_error(self, operation: str, topic: Optional[str], exc: Exception) -> None:
        """Log an exception that occurred during a Redis operation.

        Args:
            operation: Name of the operation (e.g. ``write``).
            topic: The affected topic, if any.
            exc: The caught exception.
        """
        if topic:
            logger.warning("AgentMemory.%s failed for topic %s: %s", operation, topic, exc)
        else:
            logger.warning("AgentMemory.%s failed: %s", operation, exc)

    async def _add_topic_to_set(self, topic: str) -> None:
        """Add *topic* to the Redis set tracking all known topics.

        This method is tolerant to errors; failures are logged but do not raise.

        Args:
            topic: The memory topic to register.
        """
        if not topic:
            await self._log_error("add_topic_to_set", topic, ValueError("Invalid topic"))
            return
        try:
            await self._r.sadd(_TOPICS_SET_KEY, topic)
        except Exception as e:
            await self._log_error("add_topic_to_set", topic, e)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, topic: str, data: Dict[str, Any]) -> None:
        """Append a timestamped observation to the list for *topic*.

        The list is trimmed to ``_MAX_LIST_LEN`` entries to prevent unbounded growth.

        Args:
            topic: The memory topic name.
            data: The observation payload.
        """
        if not topic:
            await self._log_error("write", topic, ValueError("Topic cannot be empty"))
            return
        payload = self._payload(data)
        key = self._key(topic)
        try:
            await self._r.lpush(key, payload)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
            await self._add_topic_to_set(topic)
        except Exception as e:
            await self._log_error("write", topic, e)

    async def set_latest(self, topic: str, data: Dict[str, Any]) -> None:
        """Overwrite the single‑latest value for *topic*.

        Args:
            topic: The memory topic name.
            data: The payload to store.
        """
        if not topic:
            await self._log_error("set_latest", topic, ValueError("Topic cannot be empty"))
            return
        payload = self._payload(data)
        key = self._key(topic, latest=True)
        try:
            await self._r.set(key, payload)
            await self._add_topic_to_set(topic)
        except Exception as e:
            await self._log_error("set_latest", topic, e)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str, n: int = 50) -> List[Dict[str, Any]]:
        """Retrieve up to *n* most‑recent observations for *topic*.

        Args:
            topic: The memory topic name.
            n: Maximum number of items to return. Must be positive.

        Returns:
            A list of deserialized observation dictionaries, newest first.
            Returns an empty list if the topic is invalid or an error occurs.
        """
        if not topic:
            await self._log_error("read_recent", topic, ValueError("Topic cannot be empty"))
            return []
        if n <= 0:
            return []
        key = self._key(topic)
        try:
            items = await self._r.lrange(key, 0, n - 1)
            return [json.loads(i) for i in items]
        except Exception as e:
            await self._log_error("read_recent", topic, e)
            return []

    async def get_latest(self, topic: str) -> Optional[Dict[str, Any]]:
        """Retrieve the latest single‑value payload for *topic*.

        Args:
            topic: The memory topic name.

        Returns:
            The deserialized payload if present, otherwise ``None``.
        """
        if not topic:
            await self._log_error("get_latest", topic, ValueError("Topic cannot be empty"))
            return None
        key = self._key(topic, latest=True)
        try:
            val = await self._r.get(key)
            return json.loads(val) if val else None
        except Exception as e:
            await self._log_error("get_latest", topic, e)
            return None

    async def read_all_topics(self) -> List[str]:
        """Return a list of all topics currently stored in memory.

        The result is cached for ``_CACHE_TTL`` seconds to reduce Redis load.

        Returns:
            A list of topic strings. Returns an empty list on error.
        """
        async with self._lock:
            now = time.time()
            if self._topics_cache is not None and (now - self._cache_timestamp) < _CACHE_TTL:
                return self._topics_cache

            try:
                raw_topics = await self._r.smembers(_TOPICS_SET_KEY)
                if not raw_topics:
                    self._topics_cache = []
                    self._cache_timestamp = now
                    return []
                # ``smembers`` may return bytes; ensure strings
                topics = [
                    t.decode() if isinstance(t, (bytes, bytearray)) else str(t)
                    for t in raw_topics
                ]
                self._topics_cache = topics
                self._cache_timestamp = now
                return topics
            except Exception as e:
                await self._log_error("read_all_topics", None, e)
                return []