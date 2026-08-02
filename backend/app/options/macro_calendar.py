"""
FOMC and macro event calendar.
Key dates sourced from Federal Reserve schedule (hardcoded 2025-2026).
Economic data via FRED API (free, no key required for basic calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Literal, Sequence


@dataclass
class MacroEvent:
    date: date
    title: str
    category: Literal["fomc", "cpi", "ppi", "nfp", "gdp", "earnings", "other"]
    importance: Literal["high", "medium", "low"]
    description: str

    def to_dict(self) -> dict:
        """Serialize the event to a JSON‑compatible dictionary."""
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


def get_upcoming_events(days_ahead: int = 90) -> list[dict]:
    """Return a list of upcoming macro events up to *days_ahead* days."""
    today = date.today()
    # Look ahead one calendar year to capture year‑end events
    cutoff = date(today.year + 1, today.month, today.day)
    events: list[MacroEvent] = []

    # Fixed FOMC dates
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

    # Approximate monthly events for the next four months
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


def get_next_fomc() -> dict | None:
    """Return the next scheduled FOMC meeting, or ``None`` if none remain."""
    today = date.today()
    for d in sorted(FOMC_2025 + FOMC_2026):
        if d >= today:
            return {
                "date": d.isoformat(),
                "days_away": (d - today).days,
                "title": "FOMC Rate Decision",
            }
    return None


# -------------------------------------------------------------------------
# Strategy helpers – signal generation & exit logic
# -------------------------------------------------------------------------

def _default_confirmation(_: MacroEvent) -> bool:
    """Fallback confirmation filter that always passes."""
    return True


def filter_events_for_signal(
    events: Sequence[MacroEvent],
    max_days_away: int = 5,
    allowed_categories: Sequence[str] | None = None,
    confirmation: Callable[[MacroEvent], bool] = _default_confirmation,
) -> list[MacroEvent]:
    """
    Tighten entry conditions by applying several filters:

    * ``max_days_away`` – only consider events occurring within the next *N* days.
    * ``allowed_categories`` – restrict signals to a whitelist of event categories.
    * ``confirmation`` – an optional callable that provides an additional
      confirmation filter (e.g., volatility, price‑action checks).

    Returns a list of events that satisfy all criteria.
    """
    if allowed_categories is None:
        allowed_categories = ["fomc", "cpi", "nfp", "gdp"]

    filtered: list[MacroEvent] = []
    today = date.today()
    for ev in events:
        days_away = (ev.date - today).days
        if ev.importance != "high":
            continue
        if days_away < 0 or days_away > max_days_away:
            continue
        if ev.category not in allowed_categories:
            continue
        if not confirmation(ev):
            continue
        filtered.append(ev)
    return filtered


def generate_entry_signals(
    days_ahead: int = 90,
    max_days_away: int = 5,
    allowed_categories: Sequence[str] | None = None,
    confirmation: Callable[[MacroEvent], bool] = _default_confirmation,
) -> list[dict]:
    """
    Produce entry signals for upcoming macro events.

    The function first fetches the raw event list, then applies the tightened
    entry filters defined in :func:`filter_events_for_signal`.  Each returned
    dictionary contains the minimal information required by the execution layer
    (date, category, and a human‑readable title).
    """
    raw_events = [
        MacroEvent(**e)  # type: ignore[arg-type] – dict keys match the dataclass fields
        for e in get_upcoming_events(days_ahead=days_ahead)
    ]
    candidates = filter_events_for_signal(
        raw_events,
        max_days_away=max_days_away,
        allowed_categories=allowed_categories,
        confirmation=confirmation,
    )
    return [
        {
            "date": ev.date.isoformat(),
            "category": ev.category,
            "title": ev.title,
            "days_away": (ev.date - date.today()).days,
        }
        for ev in candidates
    ]


def should_exit_position(event_date: date, exit_buffer_days: int = 2) -> bool:
    """
    Determine whether a position tied to a macro event should be exited.

    The default logic exits the position *exit_buffer_days* after the event
    date, allowing a short window for post‑event price reaction while avoiding
    unnecessary exposure.

    Parameters
    ----------
    event_date: date
        The calendar date of the macro event the position is linked to.
    exit_buffer_days: int, optional
        Number of days after the event to keep the position open.

    Returns
    -------
    bool
        ``True`` if the current date is beyond ``event_date + exit_buffer_days``.
    """
    today = date.today()
    return today > event_date + timedelta(days=exit_buffer_days)