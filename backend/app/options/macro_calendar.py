"""
FOMC and macro event calendar.
Key dates sourced from Federal Reserve schedule (hardcoded 2025-2026).
Economic data via FRED API (free, no key required for basic calls).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

# Constants
DEFAULT_DAYS_AHEAD: int = 90
MONTHS_AHEAD: int = 4
CPI_DAY: int = 10
PPI_DAY: int = 13
YEAR_AHEAD: int = 1

FOMC_TITLE: str = "FOMC Rate Decision"
FOMC_DESCRIPTION: str = (
    "Federal Reserve interest rate decision. Markets move ±1-2% on surprises."
)

CPI_TITLE: str = "CPI Report"
CPI_DESCRIPTION: str = "Consumer Price Index — key inflation gauge"

NFP_TITLE: str = "Non-Farm Payrolls"
NFP_DESCRIPTION: str = "Monthly jobs report — key Fed policy driver"

PPI_TITLE: str = "PPI Report"
PPI_DESCRIPTION: str = "Producer Price Index"


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


def get_upcoming_events(days_ahead: int = DEFAULT_DAYS_AHEAD) -> list[dict]:
    today = date.today()
    cutoff = date(today.year + YEAR_AHEAD, today.month, today.day)
    events: list[MacroEvent] = []

    for fomc_date in FOMC_2025 + FOMC_2026:
        if today <= fomc_date <= cutoff:
            events.append(
                MacroEvent(
                    date=fomc_date,
                    title=FOMC_TITLE,
                    category="fomc",
                    importance="high",
                    description=FOMC_DESCRIPTION,
                )
            )

    # Add approximate monthly events for next months
    for month_offset in range(MONTHS_AHEAD):
        m = ((today.month - 1 + month_offset) % 12) + 1
        y = today.year + ((today.month - 1 + month_offset) // 12)

        # CPI: ~2nd week
        cpi_date = date(y, m, CPI_DAY)
        if today <= cpi_date <= cutoff:
            events.append(
                MacroEvent(
                    date=cpi_date,
                    title=CPI_TITLE,
                    category="cpi",
                    importance="high",
                    description=CPI_DESCRIPTION,
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
                    title=NFP_TITLE,
                    category="nfp",
                    importance="high",
                    description=NFP_DESCRIPTION,
                )
            )

        # PPI: ~mid month
        ppi_date = date(y, m, PPI_DAY)
        if today <= ppi_date <= cutoff:
            events.append(
                MacroEvent(
                    date=ppi_date,
                    title=PPI_TITLE,
                    category="ppi",
                    importance="medium",
                    description=PPI_DESCRIPTION,
                )
            )

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
                "title": FOMC_TITLE,
            }
    return None