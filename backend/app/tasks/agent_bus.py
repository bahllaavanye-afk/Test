"""
Agent Message Bus — Redis Pub/Sub for real-time agent-to-agent communication.

Replaces polling-based cron jobs with event-driven agent reactions.
Every agent subscribes to topics it cares about and reacts immediately
when something relevant happens — no more waiting for the next scheduler tick.

Topics:
  market:regime       → regime changed (bull/bear/sideways)
  trade:signal        → new validated trading signal
  trade:executed      → order placed and confirmed
  trade:closed        → position exited with P&L
  research:finding    → new alpha idea from research pipeline
  strategy:updated    → strategy performance metrics changed
  experiment:done     → ML experiment completed with results
  risk:alert          → risk limit breached, circuit breaker fired
  knowledge:learned   → new lesson written to shared memory
  auction:allocated   → new capital allocation from strategy auction
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List

logger = logging.getLogger(__name__)

# All valid bus topics — unknown topics are rejected to prevent typos cascading silently
TOPICS = frozenset(
    {
        "market:regime",
        "trade:signal",
        "trade:executed",
        "trade:closed",
        "research:finding",
        "strategy:updated",
        "experiment:done",
        "risk:alert",
        "knowledge:learned",
        "auction:allocated",
    }
)

_BUS_KEY_PREFIX = "bus:events:"
_BUS_STREAM_MAX = 2000  # max events per topic (ring buffer)

Handler = Callable[[str, dict], Awaitable[None]]


class AgentBus:
    """
    Lightweight Redis-backed event bus.

    Uses Redis Streams (XADD / XREAD) instead of Pub/Sub because:
    - Streams persist messages — agents that restart don't miss events
    - Consumer groups allow multiple agents to share a topic
    - XLEN gives observable queue depth for monitoring

    Falls back gracefully when Redis is unavailable (just logs).
    """

    def __init__(self, redis_client: Any) -> None:
        self._r = redis_client
        self._handlers: Dict[str, List[Handler]] = {}
        self._consumer_offsets: Dict[str, str] = {}  # topic → last-read stream ID
        self._running = False

    # ── Publishing ────────────────────────────────────────────────────────────

    async def publish(self, topic: str, data: dict) -> None:
        """Publish an event to a topic. Fire-and-forget; never blocks callers."""
        if topic not in TOPICS:
            logger.warning("AgentBus: unknown topic %r — event dropped", topic)
            return
        payload = {
            "ts": str(time.time()),
            "topic": topic,
            **{
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in data.items()
            },
        }
        key = f"{_BUS_KEY_PREFIX}{topic}"
        try:
            await self._r.xadd(key, payload, maxlen=_BUS_STREAM_MAX, approximate=True)
        except Exception as e:  # pragma: no cover
            # Redis unavailable — log but don't crash the caller
            logger.debug("AgentBus.publish failed topic=%s: %s", topic, e)

    # ── Subscribing ───────────────────────────────────────────────────────────

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register an async handler for a topic. Called at startup, not at runtime."""
        if topic not in TOPICS:
            raise ValueError(f"AgentBus: unknown topic {topic!r}")
        self._handlers.setdefault(topic, []).append(handler)
        # Start reading from "now" — don't replay old events on (re)subscription
        self._consumer_offsets.setdefault(topic, "$")

    # ── Event loop ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the event dispatch loop as a background coroutine."""
        self._running = True
        logger.info("AgentBus: starting event loop on %d topics", len(self._handlers))
        asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        self._running = False

    async def _dispatch_loop(self) -> None:
        """Continuously poll all subscribed topics and call handlers."""
        while self._running:
            try:
                streams = self._build_streams()
                if not streams:
                    await asyncio.sleep(0.5)
                    continue

                results = await self._r.xread(streams, block=200, count=50)
                if not results:
                    continue

                await self._process_results(results)
            except Exception as exc:  # pragma: no cover
                logger.debug("AgentBus._dispatch_loop error: %s", exc)
                await asyncio.sleep(1.0)

    def _build_streams(self) -> Dict[str, str]:
        """Create the mapping of stream keys to consumer offsets for active topics."""
        return {
            f"{_BUS_KEY_PREFIX}{topic}": offset
            for topic, offset in self._consumer_offsets.items()
            if topic in self._handlers
        }

    async def _process_results(self, results: list) -> None:
        """Iterate over XREAD results, deserialize messages, and dispatch to handlers."""
        for stream_key, messages in results:
            topic = stream_key.removeprefix(_BUS_KEY_PREFIX)
            for msg_id, fields in messages:
                self._consumer_offsets[topic] = msg_id
                data = self._deserialize_fields(fields)
                await self._dispatch_to_handlers(topic, data)

    async def _dispatch_to_handlers(self, topic: str, data: dict) -> None:
        """Execute all registered handlers for a topic, protecting against exceptions."""
        for handler in self._handlers.get(topic, []):
            try:
                await handler(topic, data)
            except Exception as exc:
                logger.error(
                    "AgentBus handler error topic=%s handler=%s: %s",
                    topic,
                    getattr(handler, "__name__", repr(handler)),
                    exc,
                )

    @staticmethod
    def _deserialize_fields(fields: dict) -> dict:
        """Convert raw Redis field values back to native Python types."""
        data: dict = {}
        for k, v in fields.items():
            key = k.decode() if isinstance(k, (bytes, bytearray)) else k
            try:
                value = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                value = v.decode() if isinstance(v, (bytes, bytearray)) else v
            data[key] = value
        return data

    # ── Introspection ─────────────────────────────────────────────────────────

    async def queue_depth(self, topic: str) -> int:
        """How many events are buffered for a topic (monitoring)."""
        try:
            return await self._r.xlen(f"{_BUS_KEY_PREFIX}{topic}")
        except Exception:
            return -1


# ── Global singleton ──────────────────────────────────────────────────────────

_bus: AgentBus | None = None


def get_bus(redis_client: Any | None = None) -> AgentBus:
    global _bus
    if _bus is None:
        if redis_client is None:
            from app.redis_client import get_redis

            redis_client = get_redis()
        _bus = AgentBus(redis_client)
    return _bus