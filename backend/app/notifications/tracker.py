"""
In-memory event tracker — records all significant events with timestamps.
Used by the Slack bot and the dashboard /activity endpoint.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import islice
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class TrackedEvent:
    timestamp: datetime
    event_type: str
    category: str  # 'order' | 'signal' | 'risk' | 'experiment' | 'system'
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "category": self.category,
            "summary": self.summary,
            "metadata": self.metadata,
        }


class ActivityTracker:
    """Bounded in-memory event log (last N events). Thread‑safe via append."""

    def __init__(self, max_size: int = 5000):
        self._events: deque[TrackedEvent] = deque(maxlen=max_size)
        self._counts: dict[str, int] = {}

    def record(self, event_type: str, category: str, summary: str, **metadata) -> TrackedEvent:
        event = TrackedEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            category=category,
            summary=summary,
            metadata=metadata,
        )
        self._events.append(event)
        key = f"{category}.{event_type}"
        self._counts[key] = self._counts.get(key, 0) + 1

        # Structured logging of key metrics at INFO level
        log_payload: dict[str, Any] = {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event_type,
            "category": category,
            "summary": summary,
        }
        for metric in ("signal_count", "execution_time", "pnl"):
            if metric in metadata:
                log_payload[metric] = metadata[metric]

        logger.info("Tracked event recorded", extra=log_payload)

        return event

    def recent(self, limit: int = 100, category: str | None = None) -> List[dict]:
        """
        Return the most recent events as dictionaries.

        Args:
            limit: Maximum number of events to return.
            category: If provided, only events matching this category are returned.

        Optimized to avoid materialising the entire deque when possible.
        """
        if limit <= 0:
            return []

        # When no category filter is applied and the requested limit
        # exceeds the stored size, we can return all events directly.
        if category is None and limit >= len(self._events):
            return [e.to_dict() for e in reversed(self._events)]

        # Use a generator to avoid copying the whole deque.
        reversed_events = reversed(self._events)
        if category is not None:
            filtered = (e for e in reversed_events if e.category == category)
        else:
            filtered = reversed_events

        # islice provides an efficient early‑exit after `limit` items.
        limited = islice(filtered, limit)
        return [e.to_dict() for e in limited]

    def stats(self) -> Dict[str, Any]:
        """Return simple statistics about the stored events."""
        return {"total_events": len(self._events), "by_type": dict(self._counts)}


tracker = ActivityTracker()