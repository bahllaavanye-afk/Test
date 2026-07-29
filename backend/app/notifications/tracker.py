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

_ALLOWED_CATEGORIES = {"order", "signal", "risk", "experiment", "system"}


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
        self._events: deque[TrackedEvent] = deque(maxlen=max_size)
        self._counts: dict[str, int] = {}

    def _validate_record_params(
        self, event_type: str, category: str, summary: str, metadata: dict[str, Any]
    ) -> None:
        if not isinstance(event_type, str):
            raise TypeError("event_type must be a string")
        if not isinstance(category, str):
            raise TypeError("category must be a string")
        if not isinstance(summary, str):
            raise TypeError("summary must be a string")
        if not event_type:
            raise ValueError("event_type cannot be empty")
        if not category:
            raise ValueError("category cannot be empty")
        if not summary:
            raise ValueError("summary cannot be empty")
        if category not in _ALLOWED_CATEGORIES:
            raise ValueError(f"category '{category}' is not among allowed {_ALLOWED_CATEGORIES}")
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")

    def record(self, event_type: str, category: str, summary: str, **metadata) -> TrackedEvent:
        try:
            self._validate_record_params(event_type, category, summary, metadata)
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
            # Include optional metrics if present
            for metric in ("signal_count", "execution_time", "pnl"):
                if metric in metadata:
                    log_payload[metric] = metadata[metric]

            logger.info("Tracked event recorded", extra=log_payload)
            return event
        except Exception as exc:
            logger.error(
                "Failed to record tracked event",
                extra={
                    "event_type": event_type,
                    "category": category,
                    "summary": summary,
                    "metadata": metadata,
                    "error": str(exc),
                },
                exc_info=True,
            )
            raise

    def recent(self, limit: int = 100, category: str | None = None) -> list[dict]:
        try:
            if not isinstance(limit, int):
                raise TypeError("limit must be an integer")
            if limit <= 0:
                raise ValueError("limit must be a positive integer")
            events = list(self._events)
            if category:
                if not isinstance(category, str):
                    raise TypeError("category filter must be a string")
                events = [e for e in events if e.category == category]
            return [e.to_dict() for e in reversed(events[-limit:])]
        except Exception as exc:
            logger.error(
                "Failed to retrieve recent events",
                extra={"limit": limit, "category": category, "error": str(exc)},
                exc_info=True,
            )
            raise

    def stats(self) -> dict:
        try:
            return {"total_events": len(self._events), "by_type": dict(self._counts)}
        except Exception as exc:
            logger.error(
                "Failed to compute tracker stats",
                extra={"error": str(exc)},
                exc_info=True,
            )
            raise


tracker = ActivityTracker()