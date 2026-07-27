"""
Macro event calendar utilities.

Provides a static list of upcoming Federal Open Market Committee (FOMC) meetings
and approximated dates for key macroeconomic releases such as CPI, NFP, and PPI.
The data is hard‑coded for 2025‑2026 and does not rely on external paid APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, List, Dict, Optional


@dataclass
class MacroEvent:
    """
    Representation of a single macroeconomic event.

    Attributes
    ----------
    date: date
        The calendar date of the event.
    title: str
        Human‑readable title of the event.
    category: Literal["fomc", "cpi", "ppi", "nfp", "gdp", "earnings", "other"]
        Classification used for filtering and display.
    importance: Literal["high", "medium", "low"]
        Relative importance of the event for market participants.
    description: str
        Brief description providing context for the event.
    """

    date: date
    title: str
    category: Literal["fomc", "cpi", "ppi", "nfp", "gdp", "earnings", "other"]
    importance: Literal["high", "medium", "low"]
    description: str

    def to_dict(self) -> Dict[str, object]:
        """
        Convert the ``MacroEvent`` instance to a JSON‑serialisable dictionary.

        Returns
        -------
        dict
            Mapping containing the event fields plus a ``days_away`` key that
            indicates the number of days from today to the event date.
        """
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


def get_upcoming_events(days_ahead: int = 90) -> List[Dict[str, object]]:
    """
    Retrieve a list of upcoming macro events for the next ``days_ahead`` days.

    The function combines hard‑coded FOMC dates with approximated monthly
    economic releases (CPI, NFP, PPI) for the next four calendar months.
    Events are filtered to those occurring between today and one year from
    today, sorted chronologically, and truncated to the requested horizon.

    Parameters
    ----------
    days_ahead: int, default 90
        Maximum number of days from today to include events. The function
        returns at most ``days_ahead`` events, not days.

    Returns
    -------
    List[dict]
        A list of dictionaries, each representing an event in the format
        produced by :meth:`MacroEvent.to_dict`.
    """
    today = date.today()
    cutoff = date(today.year + 1, today.month, today.day)
    events: List[MacroEvent] = []

    for fomc_date in FOMC_2025 + FOMC_2026:
        if today <= fomc_date <= cutoff:
            events.append(
                MacroEvent(
                    date=fomc_date,
                    title="FOMC Rate Decision",
                    category="fomc",
                    importance="high",
                    description="Federal Reserve interest rate decision. Markets move ±1-2% on surprises.",
                )
            )

    # Add approximate monthly events for next 4 months
    for month_offset in range(4):
        m = ((today.month - 1 + month_offset) % 12) + 1
        y = today.year + ((today.month - 1 + month_offset) // 12)

        # CPI: ~2nd week
        cpi_date = date(y, m, 10)
        if today <= cpi_date <= cutoff:
            events.append(
                MacroEvent(
                    date=cpi_date,
                    title="CPI Report",
                    category="cpi",
                    importance="high",
                    description="Consumer Price Index — key inflation gauge",
                )
            )

        # NFP: first Friday
        first_day = date(y, m, 1)
        days_to_friday = (4 - first_day.weekday()) % 7
        nfp_date = date(y, m, 1 + days_to_friday)
        if today <= nfp_date <= cutoff:
            events.append(
                MacroEvent(
                    date=nfp_date,
                    title="Non-Farm Payrolls",
                    category="nfp",
                    importance="high",
                    description="Monthly jobs report — key Fed policy driver",
                )
            )

        # PPI: ~mid month
        ppi_date = date(y, m, 13)
        if today <= ppi_date <= cutoff:
            events.append(
                MacroEvent(
                    date=ppi_date,
                    title="PPI Report",
                    category="ppi",
                    importance="medium",
                    description="Producer Price Index",
                )
            )

    events.sort(key=lambda e: e.date)
    upcoming = [e for e in events if e.date >= today][:days_ahead]
    return [e.to_dict() for e in upcoming]


def get_next_fomc() -> Optional[Dict[str, object]]:
    """
    Return the next scheduled FOMC meeting relative to today.

    Scans the combined 2025‑2026 FOMC schedule and provides the date,
    number of days away, and a static title.

    Returns
    -------
    dict | None
        Dictionary with keys ``date``, ``days_away``, and ``title`` if a
        future meeting exists; otherwise ``None``.
    """
    today = date.today()
    for d in sorted(FOMC_2025 + FOMC_2026):
        if d >= today:
            return {
                "date": d.isoformat(),
                "days_away": (d - today).days,
                "title": "FOMC Rate Decision",
            }
    return None