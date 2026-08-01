"""
Trade Archiver: writes every order, fill, and signal to JSON‑lines files
for long‑term audit and replay. Files rotate daily.
"""
from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, TypedDict, Optional, List, Dict

from app.utils.logging import logger

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ARCHIVE_DIR: Final[Path] = Path(__file__).parents[3] / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Global lock ensures atomic writes per process; contention is low.
_lock = asyncio.Lock()

# Accepted categories – keep in sync with downstream consumers.
Category = Literal["orders", "fills", "signals", "decisions", "risk"]
VALID_CATEGORIES: Final[set[Category]] = {
    "orders",
    "fills",
    "signals",
    "decisions",
    "risk",
}


class ArchiveRecord(TypedDict, total=False):
    """Structure of a JSON‑lines record."""
    ts: str
    # Additional fields are free‑form and defined by callers.


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _today_file(category: Category) -> Path:
    """Return the Path for today's archive file of the given *category*."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ARCHIVE_DIR / f"{category}_{date_str}.jsonl"


def _sync_append(path: str, line: str) -> None:
    """Append *line* to *path* using a blocking file‑write."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as exc:
        # Propagate as a generic exception – caller logs details.
        raise RuntimeError(f"Failed to write to archive file {path}") from exc


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

async def archive_event(category: Category, data: Dict[str, object]) -> None:
    """
    Append a single JSON‑lines record to today's file for *category*.

    The function is safe for concurrent use thanks to an asyncio lock.
    Any failure is logged but does not raise to the caller, preserving
    the trading workflow.
    """
    if category not in VALID_CATEGORIES:
        logger.warning("Attempted to archive unknown category", category=category)
        return

    # Ensure a timestamp field is present.
    record: ArchiveRecord = {"ts": datetime.now(timezone.utc).isoformat(), **data}
    line = json.dumps(record, default=str) + "\n"
    file = _today_file(category)

    try:
        loop = asyncio.get_running_loop()
        async with _lock:
            await loop.run_in_executor(None, _sync_append, str(file), line)
    except Exception as exc:  # pragma: no cover
        logger.warning("Archive failed", category=category, error=str(exc))


def replay(category: Category, date_str: Optional[str] = None, limit: int = 1000) -> List[Dict[str, object]]:
    """
    Read back archived events for *category* and *date_str* (YYYY‑MM‑DD).

    Returns up to *limit* records. Corrupted lines are skipped with a debug log.
    """
    if category not in VALID_CATEGORIES:
        logger.warning("Replay requested for unknown category", category=category)
        return []

    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file = ARCHIVE_DIR / f"{category}_{date_str}.jsonl"
    if not file.exists():
        return []

    out: List[Dict[str, object]] = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
                if len(out) >= limit:
                    break
            except json.JSONDecodeError:
                logger.debug("Skipping malformed archive line", file=str(file))
                continue
    return out


def list_archives() -> Dict[str, List[str]]:
    """
    Return a mapping ``{category: [date1, date2, ...]}`` of all archive files.

    Dates are sorted chronologically for each category.
    """
    result: Dict[str, List[str]] = {}
    for f in sorted(ARCHIVE_DIR.glob("*.jsonl")):
        try:
            category, date_str = f.stem.rsplit("_", 1)
        except ValueError:
            # Unexpected file naming – ignore but log for diagnostics.
            logger.debug("Ignoring unexpected archive file", file=str(f))
            continue
        result.setdefault(category, []).append(date_str)

    # Ensure each date list is sorted (glob already sorts by filename).
    for dates in result.values():
        dates.sort()
    return result