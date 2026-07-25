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
        payload = json.dumps({"ts": time.time(), **data})
        key = f"{_PREFIX}{topic}"
        try:
            await self._r.lpush(key, payload)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
        except Exception as e:
            logger.warning("AgentMemory.write failed for topic %s: %s", topic, e)

    async def set_latest(self, topic: str, data: dict) -> None:
        """Overwrite the latest value for a topic (single-value slot)."""
        key = f"{_PREFIX}latest:{topic}"
        payload = json.dumps({"ts": time.time(), **data})
        try:
            await self._r.set(key, payload)
        except Exception as e:
            logger.warning("AgentMemory.set_latest failed for topic %s: %s", topic, e)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str, n: int = 50) -> list[dict]:
        """Return up to n most-recent observations for a topic."""
        key = f"{_PREFIX}{topic}"
        try:
            items = await self._r.lrange(key, 0, n - 1)
            return [json.loads(i) for i in items]
        except Exception as e:
            logger.warning("AgentMemory.read_recent failed for topic %s: %s", topic, e)
            return []

    async def get_latest(self, topic: str) -> dict | None:
        """Return the latest single-value for a topic."""
        key = f"{_PREFIX}latest:{topic}"
        try:
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
            return [k.removeprefix(_PREFIX) for k in keys]
        except Exception as e:
            logger.warning("AgentMemory.read_all_topics failed: %s", e)
            return []


# ----------------------------------------------------------------------
# Unit tests for edge cases (boundary conditions)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    import unittest

    class _MockRedis:
        """A minimal async in‑memory mock of the Redis interface used by AgentMemory."""
        def __init__(self):
            self._store: dict[str, list[bytes] | bytes] = {}

        async def lpush(self, key: str, value: str) -> None:
            self._store.setdefault(key, []).insert(0, value.encode())

        async def ltrim(self, key: str, start: int, end: int) -> None:
            if key in self._store and isinstance(self._store[key], list):
                self._store[key] = self._store[key][start : end + 1]

        async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
            if key not in self._store or not isinstance(self._store[key], list):
                return []
            return self._store[key][start : end + 1]

        async def set(self, key: str, value: str) -> None:
            self._store[key] = value.encode()

        async def get(self, key: str) -> bytes | None:
            val = self._store.get(key)
            return val if isinstance(val, bytes) else None

        async def keys(self, pattern: str) -> list[str]:
            # Very simple pattern handling: only supports prefix*
            prefix = pattern.rstrip("*")
            return [k for k in self._store if k.startswith(prefix)]

    class TestAgentMemoryEdgeCases(unittest.IsolatedAsyncioTestCase):
        async def asyncSetUp(self):
            self.redis = _MockRedis()
            self.mem = AgentMemory(self.redis)

        async def test_write_empty_dict(self):
            """Writing an empty dict should still store a timestamp."""
            await self.mem.write("empty_topic", {})
            recent = await self.mem.read_recent("empty_topic", n=1)
            self.assertEqual(len(recent), 1)
            self.assertIn("ts", recent[0])
            # No other keys should be present
            self.assertEqual(set(recent[0].keys()), {"ts"})

        async def test_read_recent_zero(self):
            """Reading with n=0 should return an empty list without error."""
            # Pre‑populate with some data
            await self.mem.write("some_topic", {"val": 1})
            recent = await self.mem.read_recent("some_topic", n=0)
            self.assertEqual(recent, [])

        async def test_max_list_length_enforced(self):
            """The list should never exceed _MAX_LIST_LEN items."""
            topic = "bounded_topic"
            # Insert more than the max allowed items
            for i in range(_MAX_LIST_LEN + 10):
                await self.mem.write(topic, {"idx": i})
            recent = await self.mem.read_recent(topic, n=_MAX_LIST_LEN + 20)
            self.assertEqual(len(recent), _MAX_LIST_LEN)
            # The most recent item should be the last written (i = _MAX_LIST_LEN + 9)
            self.assertEqual(recent[0]["idx"], _MAX_LIST_LEN + 9)

    # Run the tests when the module is executed directly
    unittest.main()