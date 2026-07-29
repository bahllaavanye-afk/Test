"""
In-memory event tracker — records all significant events with timestamps.
Used by the Slack bot and the dashboard /activity endpoint.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class TrackedEvent(BaseModel):
    """Schema representing a single tracked event."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the event was recorded.",
        example="2024-01-01T12:00:00Z",
    )
    event_type: str = Field(
        ...,
        description="Short identifier for the type of event.",
        example="order_filled",
    )
    category: str = Field(
        ...,
        description="High‑level classification of the event.",
        example="order",
    )
    summary: str = Field(
        ...,
        description="Human‑readable short description of the event.",
        example="Order 12345 filled at $100.00",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional data associated with the event.",
        example={"signal_count": 3, "pnl": 150.5},
    )

    @validator("category")
    def validate_category(cls, v: str) -> str:
        allowed = {"order", "signal", "risk", "experiment", "system"}
        if v not in allowed:
            raise ValueError(f"category must be one of {sorted(allowed)}")
        return v

    @validator("metadata")
    def validate_metadata_keys(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        for key in v.keys():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
        return v

    def to_dict(self) -> dict:
        """Return a JSON‑serialisable representation."""
        data = self.dict()
        data["timestamp"] = self.timestamp.isoformat()
        return data


class ActivityTracker:
    """Bounded in-memory event log (last N events). Thread‑safe via append."""

    def __init__(self, max_size: int = 5000):
        self._events: deque[TrackedEvent] = deque(maxlen=max_size)
        self._counts: Dict[str, int] = {}

    def record(self, event_type: str, category: str, summary: str, **metadata) -> TrackedEvent:
        event = TrackedEvent(
            event_type=event_type,
            category=category,
            summary=summary,
            metadata=metadata,
        )
        self._events.append(event)

        key = f"{category}.{event_type}"
        self._counts[key] = self._counts.get(key, 0) + 1

        # Structured logging of key metrics at INFO level
        log_payload: Dict[str, Any] = {
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

    def recent(self, limit: int = 100, category: str | None = None) -> list[dict]:
        events = list(self._events)
        if category:
            events = [e for e in events if e.category == category]
        return [e.to_dict() for e in reversed(events[-limit:])]

    def stats(self) -> dict:
        return {"total_events": len(self._events), "by_type": dict(self._counts)}


tracker = ActivityTracker()