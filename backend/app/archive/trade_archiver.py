"""
Trade Archiver: writes every order, fill, and signal to JSON-lines files
for long-term audit and replay. Files rotate daily.
"""
from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.logging import logger

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Minimum confidence required for a signal to be archived.
SIGNAL_CONFIDENCE_THRESHOLD: float = 0.70
# Required keys for each category; used for basic validation before archiving.
_REQUIRED_KEYS: dict[str, set[str]] = {
    "orders": {"order_id", "symbol", "side", "quantity"},
    "fills": {"fill_id", "order_id", "symbol", "side", "quantity", "price"},
    "signals": {"symbol", "side", "price", "confidence"},
    "decisions": {"decision_id", "symbol", "action"},
    "risk": {"risk_id", "metric", "value"},
}
# --------------------------------------------------------------------------- #

ARCHIVE_DIR = Path(__file__).parents[3] / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
_lock = asyncio.Lock()


def _today_file(category: str) -> Path:
    """Return the Path for today's archive file of the given category."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ARCHIVE_DIR / f"{category}_{date_str}.jsonl"


def _sync_append(path: str, line: str) -> None:
    """Synchronous file append – used inside a thread executor."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _validate_record(category: str, data: Dict[str, Any]) -> bool:
    """
    Basic validation of a record before it is archived.
    - Checks required keys for the category.
    - For signals, applies confidence threshold.
    Returns True if the record passes validation, False otherwise.
    """
    required = _REQUIRED_KEYS.get(category)
    if required is None:
        logger.warning("Unknown archive category", category=category)
        return False

    missing = required - data.keys()
    if missing:
        logger.warning(
            "Record missing required fields",
            category=category,
            missing=list(missing),
            data=data,
        )
        return False

    if category == "signals":
        confidence = data.get("confidence")
        if not isinstance(confidence, (int, float)):
            logger.warning("Signal confidence not numeric", data=data)
            return False
        if confidence < SIGNAL_CONFIDENCE_THRESHOLD:
            logger.debug(
                "Signal confidence below threshold – not archived",
                confidence=confidence,
                threshold=SIGNAL_CONFIDENCE_THRESHOLD,
                data=data,
            )
            return False

    return True


async def archive_event(category: str, data: Dict[str, Any]) -> None:
    """
    Append a single JSON line to today's file for the given category.
    The operation is atomic via an asyncio lock.

    Parameters
    ----------
    category: str
        One of 'orders', 'fills', 'signals', 'decisions', 'risk'.
    data: dict
        Event payload. Must contain the required fields for the category.
    """
    if not _validate_record(category, data):
        # Invalid record – skip archiving but keep system running.
        return

    record = {"ts": datetime.now(timezone.utc).isoformat(), **data}
    line = json.dumps(record, default=str) + "\n"
    file = _today_file(category)

    try:
        loop = asyncio.get_running_loop()
        async with _lock:
            await loop.run_in_executor(None, _sync_append, str(file), line)
    except Exception as e:
        logger.warning("Archive failed", category=category, error=str(e))


def replay(category: str, date_str: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Read back archived events for a category and date (YYYY-MM-DD).

    Parameters
    ----------
    category: str
        Archive category to read.
    date_str: str | None
        Date string in YYYY-MM-DD format. Defaults to today (UTC).
    limit: int
        Maximum number of records to return.

    Returns
    -------
    list[dict]
        List of deserialized JSON records, up to `limit`.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file = ARCHIVE_DIR / f"{category}_{date_str}.jsonl"
    if not file.exists():
        return []

    out: List[Dict[str, Any]] = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
                if len(out) >= limit:
                    break
            except json.JSONDecodeError:
                # Skip malformed lines but continue processing.
                continue
    return out


def list_archives() -> dict[str, List[str]]:
    """
    Return a mapping of {category: [date1, date2, ...]} for all archived files.
    """
    result: dict[str, List[str]] = {}
    for f in sorted(ARCHIVE_DIR.glob("*.jsonl")):
        try:
            category, date_str = f.stem.rsplit("_", 1)
        except ValueError:
            # Unexpected filename format – ignore.
            continue
        result.setdefault(category, []).append(date_str)
    return result