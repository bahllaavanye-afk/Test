"""
Trade Archiver: writes every order, fill, and signal to JSON-lines files
for long-term audit and replay. Files rotate daily.
"""
from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.utils.logging import logger

ARCHIVE_DIR = Path(__file__).parents[3] / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
_lock = asyncio.Lock()

# -------------------------------------------------------------------------
# Configuration & validation helpers
# -------------------------------------------------------------------------

VALID_CATEGORIES = {
    "orders",
    "fills",
    "signals",
    "decisions",
    "risk",
}
SignalData = dict[str, object]

MIN_SIGNAL_CONFIDENCE: float = 0.70  # only archive signals above this confidence


def _today_file(category: str) -> Path:
    """Return the Path for today's archive file of a given category."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ARCHIVE_DIR / f"{category}_{date_str}.jsonl"


def _sync_append(path: str, line: str) -> None:
    """Append a line to a file synchronously (used in a thread executor)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _validate_signal(data: SignalData) -> bool:
    """
    Validate signal data before archiving.

    Required keys:
        - 'signal': str (e.g., 'buy' or 'sell')
        - 'price': float or int
        - 'confidence': float between 0 and 1

    Returns True if the signal meets quality thresholds, otherwise False.
    """
    required_keys = {"signal", "price", "confidence"}
    missing = required_keys - data.keys()
    if missing:
        logger.debug(
            "Signal validation failed: missing keys",
            missing=missing,
        )
        return False

    confidence = data.get("confidence")
    try:
        confidence_val = float(confidence)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.debug("Signal validation failed: confidence not numeric")
        return False

    if not (0.0 <= confidence_val <= 1.0):
        logger.debug(
            "Signal validation failed: confidence out of bounds",
            confidence=confidence_val,
        )
        return False

    if confidence_val < MIN_SIGNAL_CONFIDENCE:
        logger.debug(
            "Signal validation failed: confidence below threshold",
            confidence=confidence_val,
            threshold=MIN_SIGNAL_CONFIDENCE,
        )
        return False

    return True


def _validate_decision(data: dict) -> bool:
    """
    Basic validation for decision events.

    Required keys:
        - 'type': Literal['entry', 'exit', 'adjust']
        - 'instrument': str
    """
    required_keys = {"type", "instrument"}
    missing = required_keys - data.keys()
    if missing:
        logger.debug(
            "Decision validation failed: missing keys",
            missing=missing,
        )
        return False

    decision_type = data.get("type")
    if decision_type not in {"entry", "exit", "adjust"}:
        logger.debug(
            "Decision validation failed: unknown type",
            type=decision_type,
        )
        return False

    return True


async def archive_event(category: str, data: dict) -> None:
    """
    Archive a single event.

    Parameters
    ----------
    category : str
        One of 'orders', 'fills', 'signals', 'decisions', 'risk'.
    data : dict
        Event payload. For 'signals' and 'decisions' basic validation is applied
        to tighten entry conditions and improve signal quality.

    The function is atomic and lock‑guarded.
    """
    if category not in VALID_CATEGORIES:
        logger.warning("Attempted to archive unknown category", category=category)
        return

    # Apply category‑specific quality filters
    if category == "signals":
        if not _validate_signal(data):
            logger.info("Signal discarded due to failing quality checks", data=data)
            return
    elif category == "decisions":
        if not _validate_decision(data):
            logger.info("Decision discarded due to failing validation", data=data)
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


def replay(category: str, date_str: str | None = None, limit: int = 1000) -> list[dict]:
    """Read back archived events for a category and date (YYYY-MM-DD)."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file = ARCHIVE_DIR / f"{category}_{date_str}.jsonl"
    if not file.exists():
        return []
    out: list[dict] = []
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
        # Filename format: "<category>_<YYYY-MM-DD>.jsonl"
        category, date_str = f.stem.rsplit("_", 1)
        result.setdefault(category, []).append(date_str)
    return result