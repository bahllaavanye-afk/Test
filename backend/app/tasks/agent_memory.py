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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log_success(
        self,
        action: str,
        topic: str,
        duration_seconds: float,
        signal_count: int = 0,
        pnl: float | None = None,
    ) -> None:
        """Log a structured INFO message for a successful operation."""
        extra = {
            "action": action,
            "topic": topic,
            "duration_ms": round(duration_seconds * 1000, 2),
            "signal_count": signal_count,
        }
        if pnl is not None:
            extra["pnl"] = pnl
        logger.info(
            "AgentMemory %s succeeded for topic %s",
            action,
            topic,
            extra=extra,
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, topic: str, data: dict) -> None:
        """Append an observation to a topic list with a timestamp."""
        start = time.monotonic()
        payload = json.dumps({"ts": time.time(), **data})
        key = f"{_PREFIX}{topic}"
        try:
            await self._r.lpush(key, payload)
            await self._r.ltrim(key, 0, _MAX_LIST_LEN - 1)
            duration = time.monotonic() - start
            pnl = data.get("pnl")
            self._log_success("write", topic, duration, signal_count=1, pnl=pnl)
        except Exception as e:
            logger.warning("AgentMemory.write failed for topic %s: %s", topic, e)

    async def set_latest(self, topic: str, data: dict) -> None:
        """Overwrite the latest value for a topic (single-value slot)."""
        start = time.monotonic()
        key = f"{_PREFIX}latest:{topic}"
        payload = json.dumps({"ts": time.time(), **data})
        try:
            await self._r.set(key, payload)
            duration = time.monotonic() - start
            pnl = data.get("pnl")
            self._log_success("set_latest", topic, duration, signal_count=1, pnl=pnl)
        except Exception as e:
            logger.warning("AgentMemory.set_latest failed for topic %s: %s", topic, e)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read_recent(self, topic: str, n: int = 50) -> list[dict]:
        """Return up to n most-recent observations for a topic."""
        start = time.monotonic()
        key = f"{_PREFIX}{topic}"
        try:
            items = await self._r.lrange(key, 0, n - 1)
            result = [json.loads(i) for i in items]
            duration = time.monotonic() - start
            self._log_success(
                "read_recent", topic, duration, signal_count=len(result)
            )
            return result
        except Exception as e:
            logger.warning("AgentMemory.read_recent failed for topic %s: %s", topic, e)
            return []

    async def get_latest(self, topic: str) -> dict | None:
        """Return the latest single-value for a topic."""
        start = time.monotonic()
        key = f"{_PREFIX}latest:{topic}"
        try:
            val = await self._r.get(key)
            result = json.loads(val) if val else None
            duration = time.monotonic() - start
            signal = 1 if result else 0
            pnl = result.get("pnl") if isinstance(result, dict) else None
            self._log_success(
                "get_latest", topic, duration, signal_count=signal, pnl=pnl
            )
            return result
        except Exception as e:
            logger.warning("AgentMemory.get_latest failed for topic %s: %s", topic, e)
            return None

    async def read_all_topics(self) -> list[str]:
        """List all memory topics currently stored."""
        start = time.monotonic()
        try:
            pattern = f"{_PREFIX}*"
            keys = await self._r.keys(pattern)
            result = [k.removeprefix(_PREFIX) for k in keys]
            duration = time.monotonic() - start
            self._log_success(
                "read_all_topics", "all", duration, signal_count=len(result)
            )
            return result
        except Exception as e:
            logger.warning("AgentMemory.read_all_topics failed: %s", e)
            return []