"""
In-memory event tracker — records all significant events with timestamps.
Used by the Slack bot and the dashboard /activity endpoint.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class TrackedEvent:
    timestamp: datetime
    event_type: str
    category: str        # 'order' | 'signal' | 'risk' | 'experiment' | 'system'
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
    """Bounded in-memory event log (last N events). Thread-safe via append."""

    def __init__(self, max_size: int = 5000):
        self._events: deque[TrackedEvent] = deque(maxlen=max_size)
        self._counts: dict[str, int] = {}

    def record(self, event_type: str, category: str, summary: str, **metadata) -> TrackedEvent:
        event = TrackedEvent(
            timestamp=datetime.now(UTC),
            event_type=event_type,
            category=category,
            summary=summary,
            metadata=metadata,
        )
        self._events.append(event)
        key = f"{category}.{event_type}"
        self._counts[key] = self._counts.get(key, 0) + 1
        return event

    def recent(self, limit: int = 100, category: str | None = None) -> list[dict]:
        events = list(self._events)
        if category:
            events = [e for e in events if e.category == category]
        return [e.to_dict() for e in reversed(events[-limit:])]

    def stats(self) -> dict:
        return {"total_events": len(self._events), "by_type": dict(self._counts)}


tracker = ActivityTracker()
