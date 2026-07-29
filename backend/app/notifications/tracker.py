"""
In-memory event tracker — records all significant events with timestamps.
Used by the Slack bot and the dashboard /activity endpoint.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackedEvent:
    """
    Represents a single tracked event.

    Attributes
    ----------
    timestamp : datetime
        The UTC timestamp when the event was recorded.
    event_type : str
        A short identifier for the event (e.g., ``order_filled``).
    category : str
        High‑level classification of the event. Expected values are
        ``'order'``, ``'signal'``, ``'risk'``, ``'experiment'`` or ``'system'``.
    summary : str
        Human‑readable short description of the event.
    metadata : dict[str, Any]
        Optional additional data associated with the event.
    """

    timestamp: datetime
    event_type: str
    category: str  # 'order' | 'signal' | 'risk' | 'experiment' | 'system'
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a JSON‑serialisable dictionary.

        Returns
        -------
        dict
            Mapping containing the event fields, with the timestamp rendered
            as an ISO‑8601 string.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "category": self.category,
            "summary": self.summary,
            "metadata": self.metadata,
        }


class ActivityTracker:
    """Bounded in‑memory event log (last N events). Thread‑safe via append."""

    def __init__(self, max_size: int = 5000) -> None:
        """
        Initialise a new activity tracker.

        Parameters
        ----------
        max_size : int, optional
            Maximum number of events to retain. Older events are discarded when the
            limit is reached. Defaults to 5000.
        """
        self._events: deque[TrackedEvent] = deque(maxlen=max_size)
        self._counts: Dict[str, int] = {}

    def record(
        self,
        event_type: str,
        category: str,
        summary: str,
        **metadata: Any,
    ) -> TrackedEvent:
        """
        Record a new event.

        Parameters
        ----------
        event_type : str
            Identifier for the type of event.
        category : str
            High‑level category of the event.
        summary : str
            Brief description of the event.
        **metadata : Any
            Arbitrary additional key/value pairs describing the event.

        Returns
        -------
        TrackedEvent
            The newly created event instance.
        """
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
        log_payload: Dict[str, Any] = {
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

    def recent(
        self,
        limit: int = 100,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent events.

        Parameters
        ----------
        limit : int, optional
            Maximum number of events to return. Defaults to 100.
        category : str | None, optional
            If supplied, filter events to the given category.

        Returns
        -------
        list[dict]
            List of event dictionaries ordered from newest to oldest.
        """
        events = list(self._events)
        if category:
            events = [e for e in events if e.category == category]
        return [e.to_dict() for e in reversed(events[-limit:])]

    def stats(self) -> Dict[str, Any]:
        """
        Return basic statistics about the stored events.

        Returns
        -------
        dict
            Mapping containing the total number of retained events and a count
            per ``category.event_type`` combination.
        """
        return {"total_events": len(self._events), "by_type": dict(self._counts)}


tracker = ActivityTracker()