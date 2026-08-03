"""
Trade Archiver: writes every order, fill, and signal to JSON-lines files
for long-term audit and replay. Files rotate daily.
"""
from __future__ import annotations

import json
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

from app.utils.logging import logger

ARCHIVE_DIR = Path(__file__).parents[3] / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
_lock = asyncio.Lock()

# In‑memory counters for monitoring
_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "pnl": 0.0})

# Signal quality thresholds
_MIN_CONFIDENCE = 0.6  # Minimum confidence required for entry signals
_REQUIRED_SIGNAL_FIELDS = {"signal_id", "action", "confidence"}
_REQUIRED_EXIT_FIELDS = {"reason"}


def _today_file(category: str) -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ARCHIVE_DIR / f"{category}_{date_str}.jsonl"


def _sync_append(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _validate_signal(data: dict) -> bool:
    """
    Apply tighter entry conditions and exit confirmation filters.
    Returns True if the signal passes validation, otherwise False.
    """
    # Basic required fields
    if not _REQUIRED_SIGNAL_FIELDS.issubset(data):
        logger.debug("Signal missing required fields", data=data)
        return False

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < _MIN_CONFIDENCE:
        logger.debug("Signal confidence below threshold", confidence=confidence)
        return False

    action = data.get("action")
    if action == "enter":
        # Entry signal: ensure confidence is high and required fields exist
        return True
    elif action == "exit":
        # Exit signal: require additional confirmation fields
        if not _REQUIRED_EXIT_FIELDS.issubset(data):
            logger.debug("Exit signal missing confirmation fields", data=data)
            return False
        return True
    # Other actions are allowed without extra checks
    return True


async def archive_event(category: str, data: dict) -> None:
    """
    category: 'orders' | 'fills' | 'signals' | 'decisions' | 'risk'
    Appends a single JSON line to today's file. Atomic (lock‑guarded).
    """
    # Apply signal quality filters only for signal‑related categories
    if category in {"signals", "decisions"}:
        if not _validate_signal(data):
            logger.info("Signal filtered out by validation", category=category, data=data)
            return

    start = time.monotonic()
    record = {"ts": datetime.now(timezone.utc).isoformat(), **data}
    line = json.dumps(record, default=str) + "\n"
    file = _today_file(category)
    try:
        loop = asyncio.get_running_loop()
        async with _lock:
            await loop.run_in_executor(None, _sync_append, str(file), line)

            # Update monitoring metrics
            stats = _stats[category]
            stats["count"] += 1
            pnl = data.get("pnl")
            if isinstance(pnl, (int, float)):
                stats["pnl"] += float(pnl)

        duration = time.monotonic() - start
        logger.info(
            "Archived event",
            category=category,
            count=_stats[category]["count"],
            duration=duration,
            pnl=_stats[category]["pnl"],
        )
    except Exception as e:
        logger.warning("Archive failed", category=category, error=str(e))


def replay(category: str, date_str: str | None = None, limit: int = 1000) -> list[dict]:
    """Read back archived events for a category and date (YYYY-MM-DD)."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file = ARCHIVE_DIR / f"{category}_{date_str}.jsonl"
    if not file.exists():
        return []
    out = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
                if len(out) >= limit:
                    break
            except Exception:
                continue
    return out


def list_archives() -> dict[str, list[str]]:
    """Return {category: [date1, date2, ...]} listing."""
    result: dict[str, list[str]] = {}
    for f in sorted(ARCHIVE_DIR.glob("*.jsonl")):
        category, date_str = f.stem.rsplit("_", 1)
        result.setdefault(category, []).append(date_str)
    return result