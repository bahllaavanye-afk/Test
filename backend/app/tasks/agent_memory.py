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
import asyncio
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_PREFIX = "agent:memory:"
_MAX_LIST_LEN = 500  # cap per topic to avoid unbounded growth
_TOPICS_SET_KEY = f"{_PREFIX}topics"
_CACHE_TTL = 60  # seconds

_DEFAULT_READ_RECENT_LIMIT = 50

_ERROR_TOPIC_INVALID = "Topic must be a non‑empty string"
_ERROR_TOPIC_EMPTY = "Topic cannot be empty"

# Optional import of Redis‑specific exception hierarchy.
try:
    from aioredis.exceptions import RedisError  # type: ignore
except Exception:  # pragma: no cover
    RedisError = Exception  # Fallback if aioredis is not installed.


class AgentMemory:
    """Utility class for persisting and retrieving agent observations in Redis.

    Each *topic* is stored as a Redis list (for historical observations) and,
    optionally, a separate key holding the most‑recent single value.  All payloads
    are JSON‑encoded dictionaries that include a ``ts`` timestamp field.

    The class maintains a short‑lived in‑memory cache of the known topics to avoid
    frequent round‑trips when listing them.
    """

    def __init__(self, redis_client: Any):
        """Create a new ``AgentMemory`` instance.

        Args:
            redis_client: An ``aioredis`` compatible client used for all Redis
                operations.  The exact type is not enforced to keep the class
                flexible for testing or alternative implementations.
        """
        self._r = redis_client
        self._topics_cache: Optional[List[str]] = None
        self._cache_timestamp: float = 0.0
        self._lock = asyncio.Lock()

    # ── Helper methods ────────────────────────────────────────────────────────

    def _key(self, topic: str, *, latest: bool = False) -> str:
        """Construct a Redis key for a given topic.

        Args:
            topic: The logical name of the topic.
            latest: If ``True``, return the key for the single‑value slot;
                otherwise return the list key.

        Returns:
            The full Redis key string.

        Raises:
            ValueError: If ``topic`` is empty or falsy.
        """
        if not topic:
            raise ValueError(_ERROR_TOPIC_INVALID)
        prefix = f"{_PREFIX}latest:" if latest else _PREFIX
        return f"{prefix}{topic}"

    def _payload(self, data: Dict[str, Any]) -> str:
        """Serialise ``data`` with an embedded timestamp.

        Args:
            data: The observation payload to store.

        Returns:
            A JSON string containing the original data plus a ``ts`` field.
        """
        if data is None:
            data = {}
        return json.dumps({"ts": time.time(), **data})

    async def _log_error(self, operation: str, topic: Optional[str], exc: Exception) -> None:
        """Log a Redis operation failure with structured context.

        Args:
            operation: The name of the operation that raised the exception.
            topic: The related topic, if any.
            exc: The exception that was caught.
        """
        extra = {
            "operation": operation,
            "topic": topic,
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
        }
        logger.error("AgentMemory operation failed", extra=extra)

    async def _add_topic_to_set(self, topic: str) -> None:
        """Ensure the topic is recorded in the Redis set of topics.

        Args:
            topic: The topic name to add.
        """
        if not topic:
            await self._log_error("add_topic_to_set", topic, ValueError(_ERROR_TOPIC_INVALID))
            return
        try:
            await self._r.sadd(_TOPICS_SET_KEY, topic)
        except RedisError as e:
            await self._log_error("add_topic_to_set", topic, e)
        except Exception as e:  # pragma: no cover
            await self._log_error("add_topic_to_set", topic, e)

    def _decode_topics(self, raw_topics: Any) -> List[str]:
        """Convert raw Redis topic members to a list of strings.

        Args:
            raw_topics: Iterable of bytes or strings returned by Redis.

        Returns:
            A list of topic names as strings.
        """
        return [
            t.decode() if isinstance(t, (bytes, bytearray)) else str(t)
            for t in raw_topics
        ]

    def _update_topics_cache(self, topics: List[str]) -> None:
        """Refresh the in‑memory cache of topics.

        Args:
            topics: The list of topic names to cache.
        """
        self._topics_cache = topics
        self._cache_timestamp = time.time()

    async def _fetch_topics_from_redis(self) -> List[str]:
        """Retrieve the set of topics from Redis, handling empty results.

        Returns:
            A list of topic names.
        """
        raw_topics = await self._r.smembers(_TOPICS_SET_KEY)
        if not raw_topics:
            return []
        return self._decode_topics(raw_topics)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, topic: str, data: Dict[str, Any]) -> None:
        """Append an observation to a topic list with a timestamp.

        Args:
            topic: The logical name of the topic.
            data: The observation payload to store.
        """
        if not topic:
            await self._log_error("write", topic, ValueError(_ERROR_TOPIC_EMPTY))
            return
        payload = self._payload(data)
        key = self._key(topic)
        try:
            await self._r.lpush(key, payload)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
            await self._add_topic_to_set(topic)
        except RedisError as e:
            await self._log_error("write", topic, e)
        except Exception as e:  # pragma: no cover
            await self._log_error("write", topic, e)

    async def set_latest(self, topic: str, data: Dict[str, Any]) -> None:
        """Overwrite the latest value for a topic (single‑value slot).

        Args:
            topic: The logical name of the topic.
            data: The observation payload to store.
        """
        if not topic:
            await self._log_error("set_latest", topic, ValueError(_ERROR_TOPIC_EMPTY))
            return
        payload = self._payload(data)
        key = self._key(topic, latest=True)
        try:
            await self._r.set(key, payload)
            await self._add_topic_to_set(topic)
        except RedisError as e:
            await self._log_error("set_latest", topic, e)
        except Exception as e:  # pragma: no cover
            await self._log_error("set_latest", topic, e)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str, n: int = _DEFAULT_READ_RECENT_LIMIT) -> List[Dict[str, Any]]:
        """Return up to *n* most‑recent observations for a topic.

        Args:
            topic: The logical name of the topic.
            n: Maximum number of items to retrieve. If ``n`` is non‑positive,
               an empty list is returned.

        Returns:
            A list of observation dictionaries ordered from newest to oldest.
        """
        if not topic:
            await self._log_error("read_recent", topic, ValueError(_ERROR_TOPIC_EMPTY))
            return []
        if n <= 0:
            return []
        key = self._key(topic)
        try:
            items = await self._r.lrange(key, 0, n - 1)
            return [json.loads(i) for i in items]
        except RedisError as e:
            await self._log_error("read_recent", topic, e)
            return []
        except Exception as e:  # pragma: no cover
            await self._log_error("read_recent", topic, e)
            return []

    async def get_latest(self, topic: str) -> Optional[Dict[str, Any]]:
        """Return the latest single‑value for a topic.

        Args:
            topic: The logical name of the topic.

        Returns:
            The most recent observation dictionary, or ``None`` if no value exists.
        """
        if not topic:
            await self._log_error("get_latest", topic, ValueError(_ERROR_TOPIC_EMPTY))
            return None
        key = self._key(topic, latest=True)
        try:
            val = await self._r.get(key)
            return json.loads(val) if val else None
        except RedisError as e:
            await self._log_error("get_latest", topic, e)
            return None
        except Exception as e:  # pragma: no cover
            await self._log_error("get_latest", topic, e)
            return None

    async def read_all_topics(self) -> List[str]:
        """List all memory topics currently stored, using a short‑lived cache.

        Returns:
            A list of topic names. If the cache is fresh, the cached list is
            returned; otherwise the method queries Redis.
        """
        async with self._lock:
            now = time.time()
            if self._topics_cache is not None and (now - self._cache_timestamp) < _CACHE_TTL:
                return self._topics_cache

            try:
                topics = await self._fetch_topics_from_redis()
                self._update_topics_cache(topics)
                return topics
            except RedisError as e:
                await self._log_error("read_all_topics", None, e)
                return []
            except Exception as e:  # pragma: no cover
                await self._log_error("read_all_topics", None, e)
                return []