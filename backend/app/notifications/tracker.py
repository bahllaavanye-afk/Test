"""
In-memory event tracker — records all significant events with timestamps.
Used by the Slack bot and the dashboard /activity endpoint.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

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
    """Bounded in-memory event log (last N events). Thread-safe via append."""

    def __init__(self, max_size: int = 5000):
        # Guard against non‑positive max_size which would create an unbounded deque
        if not isinstance(max_size, int) or max_size <= 0:
            logger.warning(
                "Invalid max_size %s supplied to ActivityTracker; falling back to default 5000",
                max_size,
            )
            max_size = 5000
        self._events: deque[TrackedEvent] = deque(maxlen=max_size)
        self._counts: dict[str, int] = {}

    def _log_event(self, event: TrackedEvent) -> None:
        """
        Emit a structured log record for a tracked event.

        Includes key metrics (signal_count, execution_time, pnl) even if they are absent,
        defaulting to zero values.
        """
        base_payload: dict[str, Any] = {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "category": event.category,
            "summary": event.summary,
        }

        # Ensure key metrics are always present
        metrics = {
            "signal_count": event.metadata.get("signal_count", 0),
            "execution_time": event.metadata.get("execution_time", 0),
            "pnl": event.metadata.get("pnl", 0.0),
        }
        base_payload.update(metrics)

        # Attach the payload under a distinct attribute to avoid clashes with LogRecord fields
        logger.info("Tracked event recorded", extra={"event": base_payload})

    def record(
        self, event_type: str | None, category: str | None, summary: str | None, **metadata
    ) -> TrackedEvent:
        """
        Record a new event.

        Handles None values by substituting sensible defaults to avoid crashes.
        """
        # Defensive defaults for mandatory string fields
        event_type = event_type or "unknown_event"
        category = category or "unknown_category"
        summary = summary or ""

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

        # Structured logging of the event
        self._log_event(event)

        return event

    def recent(self, limit: int | None = 100, category: str | None = None) -> list[dict]:
        """
        Return the most recent events as dictionaries.

        - Handles None or non‑positive limits by falling back to the default.
        - Safely works when the internal collection is empty.
        - Guarantees the slice does not exceed the collection size (off‑by‑one safe).
        """
        # Validate limit
        if not isinstance(limit, int) or limit <= 0:
            logger.debug(
                "Invalid limit %s supplied to recent(); using default of 100", limit
            )
            limit = 100

        events = list(self._events)  # snapshot for thread‑safety
        if category:
            events = [e for e in events if e.category == category]

        # Slice safely: min(limit, len(events)) prevents off‑by‑one errors
        slice_end = min(limit, len(events))
        recent_slice = events[-slice_end:] if slice_end else []

        # Return in reverse chronological order
        return [e.to_dict() for e in reversed(recent_slice)]

    def stats(self) -> dict:
        """Return simple statistics about the tracked events."""
        return {"total_events": len(self._events), "by_type": dict(self._counts)}


tracker = ActivityTracker()