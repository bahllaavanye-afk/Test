"""
FOMC and macro event calendar.
Key dates sourced from Federal Reserve schedule (hardcoded 2025-2026).
Economic data via FRED API (free, no key required for basic calls).
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal

# Constants
DEFAULT_DAYS_AHEAD = 90
FUTURE_YEAR_OFFSET = 1
MONTH_LOOKAHEAD = 4

CPI_DAY_OF_MONTH = 10
PPI_DAY_OF_MONTH = 13
NFP_WEEKDAY_INDEX = 4  # Friday (0=Monday)

FOMC_EVENT_TITLE = "FOMC Rate Decision"
FOMC_EVENT_DESCRIPTION = (
    "Federal Reserve interest rate decision. Markets move ±1-2% on surprises."
)

CPI_EVENT_TITLE = "CPI Report"
CPI_EVENT_DESCRIPTION = "Consumer Price Index — key inflation gauge"

NFP_EVENT_TITLE = "Non-Farm Payrolls"
NFP_EVENT_DESCRIPTION = "Monthly jobs report — key Fed policy driver"

PPI_EVENT_TITLE = "PPI Report"
PPI_EVENT_DESCRIPTION = "Producer Price Index"


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
    # CPI releases
    ("CPI Report", "cpi", "high", "Consumer Price Index — key inflation gauge. High prints → rate hike risk"),
    # NFP
    ("Non-Farm Payrolls", "nfp", "high", "Jobs report — first Friday of month. Drives Fed policy expectations"),
    # PPI
    ("PPI Report", "ppi", "medium", "Producer Price Index — leading CPI indicator"),
    # GDP
    ("GDP (Advance)", "gdp", "high", "Quarterly GDP advance estimate"),
]


def get_upcoming_events(days_ahead: int = DEFAULT_DAYS_AHEAD) -> list[dict]:
    today = date.today()
    cutoff = date(today.year + FUTURE_YEAR_OFFSET, today.month, today.day)
    events: list[MacroEvent] = []

    for fomc_date in FOMC_2025 + FOMC_2026:
        if today <= fomc_date <= cutoff:
            events.append(
                MacroEvent(
                    date=fomc_date,
                    title=FOMC_EVENT_TITLE,
                    category="fomc",
                    importance="high",
                    description=FOMC_EVENT_DESCRIPTION,
                )
            )

    # Add approximate monthly events for next months
    import calendar

    for month_offset in range(MONTH_LOOKAHEAD):
        m = ((today.month - 1 + month_offset) % 12) + 1
        y = today.year + ((today.month - 1 + month_offset) // 12)

        # CPI: ~2nd week
        cpi_date = date(y, m, CPI_DAY_OF_MONTH)
        if today <= cpi_date <= cutoff:
            events.append(
                MacroEvent(
                    date=cpi_date,
                    title=CPI_EVENT_TITLE,
                    category="cpi",
                    importance="high",
                    description=CPI_EVENT_DESCRIPTION,
                )
            )

        # NFP: first Friday
        first_day = date(y, m, 1)
        days_to_friday = (NFP_WEEKDAY_INDEX - first_day.weekday()) % 7
        nfp_date = date(y, m, 1 + days_to_friday)
        if today <= nfp_date <= cutoff:
            events.append(
                MacroEvent(
                    date=nfp_date,
                    title=NFP_EVENT_TITLE,
                    category="nfp",
                    importance="high",
                    description=NFP_EVENT_DESCRIPTION,
                )
            )

        # PPI: ~mid month
        ppi_date = date(y, m, PPI_DAY_OF_MONTH)
        if today <= ppi_date <= cutoff:
            events.append(
                MacroEvent(
                    date=ppi_date,
                    title=PPI_EVENT_TITLE,
                    category="ppi",
                    importance="medium",
                    description=PPI_EVENT_DESCRIPTION,
                )
            )

    events.sort(key=lambda e: e.date)
    # Return only upcoming (not past)
    upcoming = [e for e in events if e.date >= today][:days_ahead]
    return [e.to_dict() for e in upcoming]


def get_next_fomc() -> dict | None:
    today = date.today()
    for d in sorted(FOMC_2025 + FOMC_2026):
        if d >= today:
            return {
                "date": d.isoformat(),
                "days_away": (d - today).days,
                "title": FOMC_EVENT_TITLE,
            }
    return None