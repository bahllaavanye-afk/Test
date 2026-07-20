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
from json import JSONDecodeError
from typing import Any

# Attempt to import RedisError for more specific exception handling.
# Fallback to generic Exception if the specific class is unavailable.
try:
    from redis.exceptions import RedisError  # type: ignore
except Exception:  # pragma: no cover
    RedisError = Exception  # type: ignore

logger = logging.getLogger(__name__)

_PREFIX = "agent:memory:"
_MAX_LIST_LEN = 500  # cap per topic to avoid unbounded growth


class AgentMemory:
    def __init__(self, redis_client: Any):
        self._r = redis_client

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, topic: str, data: dict) -> None:
        """Append an observation to a topic list with a timestamp."""
        try:
            payload = json.dumps({"ts": time.time(), **data})
        except (TypeError, ValueError) as e:
            logger.error(
                "AgentMemory.write JSON serialization failed for topic %s: %s",
                topic,
                e,
            )
            return

        key = f"{_PREFIX}{topic}"
        try:
            await self._r.lpush(key, payload)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
        except RedisError as e:
            logger.error(
                "AgentMemory.write Redis operation failed for topic %s: %s",
                topic,
                e,
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "AgentMemory.write unexpected error for topic %s: %s",
                topic,
                e,
            )

    async def set_latest(self, topic: str, data: dict) -> None:
        """Overwrite the latest value for a topic (single-value slot)."""
        try:
            payload = json.dumps({"ts": time.time(), **data})
        except (TypeError, ValueError) as e:
            logger.error(
                "AgentMemory.set_latest JSON serialization failed for topic %s: %s",
                topic,
                e,
            )
            return

        key = f"{_PREFIX}latest:{topic}"
        try:
            await self._r.set(key, payload)
        except RedisError as e:
            logger.error(
                "AgentMemory.set_latest Redis operation failed for topic %s: %s",
                topic,
                e,
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "AgentMemory.set_latest unexpected error for topic %s: %s",
                topic,
                e,
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str, n: int = 50) -> list[dict]:
        """Return up to n most-recent observations for a topic."""
        key = f"{_PREFIX}{topic}"
        try:
            items = await self._r.lrange(key, 0, n - 1)
            return [json.loads(i) for i in items]
        except RedisError as e:
            logger.error(
                "AgentMemory.read_recent Redis operation failed for topic %s: %s",
                topic,
                e,
            )
        except JSONDecodeError as e:
            logger.error(
                "AgentMemory.read_recent JSON decode failed for topic %s: %s",
                topic,
                e,
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "AgentMemory.read_recent unexpected error for topic %s: %s",
                topic,
                e,
            )
        return []

    async def get_latest(self, topic: str) -> dict | None:
        """Return the latest single-value for a topic."""
        key = f"{_PREFIX}latest:{topic}"
        try:
            val = await self._r.get(key)
            return json.loads(val) if val else None
        except RedisError as e:
            logger.error(
                "AgentMemory.get_latest Redis operation failed for topic %s: %s",
                topic,
                e,
            )
        except JSONDecodeError as e:
            logger.error(
                "AgentMemory.get_latest JSON decode failed for topic %s: %s",
                topic,
                e,
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "AgentMemory.get_latest unexpected error for topic %s: %s",
                topic,
                e,
            )
        return None

    async def read_all_topics(self) -> list[str]:
        """List all memory topics currently stored."""
        try:
            pattern = f"{_PREFIX}*"
            keys = await self._r.keys(pattern)
            return [k.removeprefix(_PREFIX) for k in keys]
        except RedisError as e:
            logger.error(
                "AgentMemory.read_all_topics Redis operation failed: %s", e
            )
        except Exception as e:  # pragma: no cover
            logger.error(
                "AgentMemory.read_all_topics unexpected error: %s", e
            )
        return []