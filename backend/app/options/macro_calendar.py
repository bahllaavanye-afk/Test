"""
FOMC and macro event calendar.
Key dates sourced from Federal Reserve schedule (hardcoded 2025-2026).
Economic data via FRED API (free, no key required for basic calls).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass
class MacroEvent:
    date: date
    title: str
    category: Literal["fomc", "cpi", "ppi", "nfp", "gdp", "earnings", "other"]
    importance: Literal["high", "medium", "low"]
    description: str

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "title": self.title,
            "category": self.category,
            "importance": self.importance,
            "description": self.description,
            "days_away": (self.date - date.today()).days,
        }


# 2025-2026 FOMC meeting schedule (dates from federalreserve.gov)
FOMC_2025 = [
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
]
FOMC_2026 = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]

# Monthly economic releases (approximate dates — varies monthly)
MONTHLY_EVENTS_2025 = [
    ("CPI Report", "cpi", "high", "Consumer Price Index — key inflation gauge. High prints → rate hike risk"),
    ("Non-Farm Payrolls", "nfp", "high", "Jobs report — first Friday of month. Drives Fed policy expectations"),
    ("PPI Report", "ppi", "medium", "Producer Price Index — leading CPI indicator"),
    ("GDP (Advance)", "gdp", "high", "Quarterly GDP advance estimate"),
]


def _first_friday(year: int, month: int) -> date:
    """Return the date of the first Friday of a given month."""
    first_day = date(year, month, 1)
    days_to_friday = (4 - first_day.weekday()) % 7
    return date(year, month, 1 + days_to_friday)


def _add_event(events: list[MacroEvent], *, event_date: date, title: str,
               category: Literal["fomc", "cpi", "ppi", "nfp", "gdp", "earnings", "other"],
               importance: Literal["high", "medium", "low"], description: str,
               today: date, cutoff: date) -> None:
    """Append a MacroEvent to the list if it falls within the allowed window."""
    if today <= event_date <= cutoff:
        events.append(
            MacroEvent(
                date=event_date,
                title=title,
                category=category,
                importance=importance,
                description=description,
            )
        )


def _generate_fomc_events(today: date, cutoff: date) -> list[MacroEvent]:
    """Create FOMC events that lie between today and cutoff."""
    events: list[MacroEvent] = []
    for fomc_date in FOMC_2025 + FOMC_2026:
        _add_event(
            events,
            event_date=fomc_date,
            title="FOMC Rate Decision",
            category="fomc",
            importance="high",
            description="Federal Reserve interest rate decision. Markets move ±1-2% on surprises.",
            today=today,
            cutoff=cutoff,
        )
    return events


def _generate_monthly_events(today: date, cutoff: date) -> list[MacroEvent]:
    """Generate approximate monthly macro events for the next four months."""
    events: list[MacroEvent] = []
    for month_offset in range(4):
        month = ((today.month - 1 + month_offset) % 12) + 1
        year = today.year + ((today.month - 1 + month_offset) // 12)

        # CPI: approx. 10th day of month
        _add_event(
            events,
            event_date=date(year, month, 10),
            title="CPI Report",
            category="cpi",
            importance="high",
            description="Consumer Price Index — key inflation gauge",
            today=today,
            cutoff=cutoff,
        )

        # NFP: first Friday of month
        _add_event(
            events,
            event_date=_first_friday(year, month),
            title="Non-Farm Payrolls",
            category="nfp",
            importance="high",
            description="Monthly jobs report — key Fed policy driver",
            today=today,
            cutoff=cutoff,
        )

        # PPI: approx. 13th day of month
        _add_event(
            events,
            event_date=date(year, month, 13),
            title="PPI Report",
            category="ppi",
            importance="medium",
            description="Producer Price Index",
            today=today,
            cutoff=cutoff,
        )
    return events


def get_upcoming_events(days_ahead: int = 90) -> list[dict]:
    """Return a list of upcoming macro events, limited to `days_ahead` entries."""
    today = date.today()
    cutoff = date(today.year + 1, today.month, today.day)

    # Collect events from both FOMC schedule and monthly approximations
    events = _generate_fomc_events(today, cutoff) + _generate_monthly_events(today, cutoff)

    # Sort chronologically and slice to the requested number of events
    events.sort(key=lambda e: e.date)
    upcoming = [e for e in events if e.date >= today][:days_ahead]
    return [e.to_dict() for e in upcoming]


def get_next_fomc() -> dict | None:
    today = date.today()
    for d in sorted(FOMC_2025 + FOMC_2026):
        if d >= today:
            return {
                "date": d.isoformat(),
                "days_away": (d - today).days,
                "title": "FOMC Rate Decision",
            }
    return None