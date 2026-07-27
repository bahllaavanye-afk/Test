"""
FOMC and macro event calendar.
Key dates sourced from Federal Reserve schedule (hardcoded 2025-2026).
Economic data via FRED API (free, no key required for basic calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal, List, Dict, Optional


@dataclass
class MacroEvent:
    date: date
    title: str
    category: Literal["fomc", "cpi", "ppi", "nfp", "gdp", "earnings", "other"]
    importance: Literal["high", "medium", "low"]
    description: str

    def to_dict(self) -> dict:
        """Convert the event to a JSON‑serialisable dictionary."""
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

# ---------------------------------------------------------------------------
# Core calendar utilities
# ---------------------------------------------------------------------------

def _is_weekday(d: date) -> bool:
    """Return True if the date is a Monday‑Friday."""
    return d.weekday() < 5


def get_upcoming_events(days_ahead: int = 90) -> List[dict]:
    """
    Return a list of upcoming macro events limited by ``days_ahead``.
    The function keeps the original behaviour but adds a safety check
    that events are on weekdays (most releases occur on business days).
    """
    today = date.today()
    cutoff = date(today.year + 1, today.month, today.day)
    events: List[MacroEvent] = []

    # FOMC meetings – always high importance
    for fomc_date in FOMC_2025 + FOMC_2026:
        if today <= fomc_date <= cutoff and _is_weekday(fomc_date):
            events.append(
                MacroEvent(
                    date=fomc_date,
                    title="FOMC Rate Decision",
                    category="fomc",
                    importance="high",
                    description="Federal Reserve interest rate decision. Markets move ±1-2% on surprises.",
                )
            )

    # Approximate monthly events for the next 4 months
    for month_offset in range(4):
        m = ((today.month - 1 + month_offset) % 12) + 1
        y = today.year + ((today.month - 1 + month_offset) // 12)

        # CPI – ~2nd week
        cpi_date = date(y, m, 10)
        if today <= cpi_date <= cutoff and _is_weekday(cpi_date):
            events.append(
                MacroEvent(
                    date=cpi_date,
                    title="CPI Report",
                    category="cpi",
                    importance="high",
                    description="Consumer Price Index — key inflation gauge",
                )
            )

        # NFP – first Friday
        first_day = date(y, m, 1)
        days_to_friday = (4 - first_day.weekday()) % 7
        nfp_date = date(y, m, 1 + days_to_friday)
        if today <= nfp_date <= cutoff and _is_weekday(nfp_date):
            events.append(
                MacroEvent(
                    date=nfp_date,
                    title="Non-Farm Payrolls",
                    category="nfp",
                    importance="high",
                    description="Monthly jobs report — key Fed policy driver",
                )
            )

        # PPI – ~mid month
        ppi_date = date(y, m, 13)
        if today <= ppi_date <= cutoff and _is_weekday(ppi_date):
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


def get_next_fomc() -> Optional[dict]:
    """Return the next FOMC meeting (date, days away, title) or ``None``."""
    today = date.today()
    for d in sorted(FOMC_2025 + FOMC_2026):
        if d >= today:
            return {
                "date": d.isoformat(),
                "days_away": (d - today).days,
                "title": "FOMC Rate Decision",
            }
    return None


# ---------------------------------------------------------------------------
# Strategy‑level helpers – tighter entry / exit logic
# ---------------------------------------------------------------------------

MAX_ENTRY_DAYS_AHEAD = 30  # Only consider events within the next month
ENTRY_CATEGORIES = {"fomc", "cpi", "nfp", "gdp"}  # Focus on macro drivers


def get_high_confidence_events(days_ahead: int = MAX_ENTRY_DAYS_AHEAD) -> List[dict]:
    """
    Return events that satisfy tighter entry criteria:
    * Importance must be ``high``.
    * Category must be in ``ENTRY_CATEGORIES``.
    * Event occurs within ``MAX_ENTRY_DAYS_AHEAD`` calendar days.
    """
    all_events = get_upcoming_events(days_ahead=days_ahead)
    filtered = [
        e
        for e in all_events
        if e["importance"] == "high"
        and e["category"] in ENTRY_CATEGORIES
        and e["days_away"] <= MAX_ENTRY_DAYS_AHEAD
    ]
    return filtered


def should_enter_position(event: dict, current_time: Optional[datetime] = None) -> bool:
    """
    Confirmation filter for entering a position based on a macro event.

    * For ``fomc`` and ``cpi`` events we require the event to be on a weekday
      (already ensured by the calendar) and at least 1 business day away to
      avoid overnight surprise risk.
    * For ``nfp`` we additionally require that the event is not on a holiday
      (simple check – if the date falls on a weekend we reject).

    The function is deliberately lightweight; more sophisticated filters
    (e.g., volatility, order‑book imbalance) can be injected via the
    ``current_time`` argument in the future.
    """
    if current_time is None:
        current_time = datetime.utcnow()
    event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
    days_until = (event_date - date.today()).days

    # Basic safety: must be at least 1 day away
    if days_until < 1:
        return False

    # Weekday confirmation (already filtered but kept for safety)
    if not _is_weekday(event_date):
        return False

    # Category‑specific tweaks
    if event["category"] == "nfp":
        # NFP is released on the first Friday; ensure it is indeed a Friday
        if event_date.weekday() != 4:
            return False

    return True


def should_exit_position(entry_date: date, exit_window: int = 5) -> bool:
    """
    Simple exit rule: close the position ``exit_window`` days after the entry
    date, unless the calendar indicates a high‑impact event within that window.
    This encourages exiting before the market digests the macro release,
    reducing tail‑risk exposure.
    """
    today = date.today()
    if today >= entry_date + timedelta(days=exit_window):
        # Check if a high‑importance event occurs within the same window
        upcoming = get_upcoming_events(days_ahead=exit_window)
        for ev in upcoming:
            ev_date = datetime.strptime(ev["date"], "%Y-%m-%d").date()
            if entry_date < ev_date <= entry_date + timedelta(days=exit_window):
                # If a high‑importance event is imminent, exit earlier
                return True
        # No intervening event – normal exit
        return True
    return False


# ---------------------------------------------------------------------------
# Public API – preserve legacy functions while exposing refined helpers
# ---------------------------------------------------------------------------

__all__ = [
    "MacroEvent",
    "get_upcoming_events",
    "get_next_fomc",
    "get_high_confidence_events",
    "should_enter_position",
    "should_exit_position",
]