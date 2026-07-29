"""
Trade Archiver: writes every order, fill, and signal to JSON‑lines files
for long-term audit and replay. Files rotate daily.
"""
from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.utils.logging import logger

ARCHIVE_DIR = Path(__file__).parents[3] / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
_lock = asyncio.Lock()


def _today_file(category: str) -> Path:
    """Return the Path for today's archive file for a given category."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ARCHIVE_DIR / f"{category}_{date_str}.jsonl"


def _sync_append(path: str, line: str) -> None:
    """Append a line to a file synchronously (used in an executor)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _validate_signal(data: Mapping[str, Any]) -> bool:
    """
    Basic sanity‑check for a signal before archiving.

    Required keys:
        - symbol: non‑empty string
        - signal_type: either "enter" or "exit"
        - confidence: float in [0, 1]

    Additional checks:
        * For entry signals, confidence must be >= 0.6.
        * For exit signals, an ``exit_reason`` key must be present.
    """
    required_keys = {"symbol", "signal_type", "confidence"}
    if not required_keys.issubset(data):
        logger.debug(
            "Signal missing required keys",
            missing=required_keys - set(data),
            data=data,
        )
        return False

    symbol = data.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        logger.debug("Invalid symbol in signal", symbol=symbol, data=data)
        return False

    signal_type = data.get("signal_type")
    if signal_type not in {"enter", "exit"}:
        logger.debug("Invalid signal_type", signal_type=signal_type, data=data)
        return False

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        logger.debug("Invalid confidence value", confidence=confidence, data=data)
        return False

    if signal_type == "enter" and confidence < 0.6:
        logger.debug(
            "Entry signal confidence too low", confidence=confidence, data=data
        )
        return False

    if signal_type == "exit" and "exit_reason" not in data:
        logger.debug("Exit signal missing exit_reason", data=data)
        return False

    return True


async def archive_event(category: str, data: dict) -> None:
    """
    Archive a single event for a given category.

    Parameters
    ----------
    category : str
        One of 'orders', 'fills', 'signals', 'decisions', 'risk'.
    data : dict
        Event payload. For ``signals`` the payload is validated and filtered;
        only high‑quality signals are persisted.

    The function is atomic thanks to a lock and runs the file write in a thread
    pool executor to avoid blocking the event loop.
    """
    # Apply confirmation filter for signals
    if category == "signals" and not _validate_signal(data):
        logger.warning("Signal failed validation and will not be archived", data=data)
        return

    record = {"ts": datetime.now(timezone.utc).isoformat(), **data}
    line = json.dumps(record, default=str) + "\n"
    file = _today_file(category)

    try:
        loop = asyncio.get_running_loop()
        async with _lock:
            await loop.run_in_executor(None, _sync_append, str(file), line)
    except Exception as e:  # pragma: no cover
        logger.warning("Archive failed", category=category, error=str(e))


def replay(category: str, date_str: str | None = None, limit: int = 1000) -> list[dict]:
    """Read back archived events for a category and date (YYYY‑MM‑DD)."""
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
                # Corrupted line – skip silently
                continue
    return out


def list_archives() -> dict[str, list[str]]:
    """Return a mapping of ``category`` → list of dates for which archives exist."""
    result: dict[str, list[str]] = {}
    for f in sorted(ARCHIVE_DIR.glob("*.jsonl")):
        # Filename pattern: <category>_YYYY‑MM‑DD.jsonl
        try:
            category, date_str = f.stem.rsplit("_", 1)
        except ValueError:
            continue
        result.setdefault(category, []).append(date_str)
    return result