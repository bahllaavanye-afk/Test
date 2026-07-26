"""
Trade Archiver: writes every order, fill, and signal to JSON-lines files
for long-term audit and replay. Files rotate daily.
"""
from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from app.utils.logging import logger

ARCHIVE_DIR = Path(__file__).parents[3] / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

_lock = asyncio.Lock()


def _today_file(category: str) -> Path:
    """Return the Path for today's archive file of the given category."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ARCHIVE_DIR / f"{category}_{date_str}.jsonl"


def _sync_append(path: str, line: str) -> None:
    """Append a line to a file synchronously."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _build_record(data: dict) -> dict:
    """Create a record with a timestamp merged into the provided data."""
    return {"ts": datetime.now(timezone.utc).isoformat(), **data}


def _serialize_record(record: dict) -> str:
    """Serialize a record to a JSON line, ensuring a trailing newline."""
    return json.dumps(record, default=str) + "\n"


async def _write_line(file: Path, line: str) -> None:
    """Write a line to the given file using the shared asyncio lock."""
    loop = asyncio.get_running_loop()
    async with _lock:
        await loop.run_in_executor(None, _sync_append, str(file), line)


async def archive_event(category: str, data: dict) -> None:
    """
    Archive a single event.

    Parameters
    ----------
    category: str
        One of 'orders', 'fills', 'signals', 'decisions', 'risk'.
    data: dict
        Event payload to be stored.

    The function builds a timestamped record, serializes it, and appends it
    atomically to today's file for the given category.
    """
    try:
        record = _build_record(data)
        line = _serialize_record(record)
        file = _today_file(category)
        await _write_line(file, line)
    except Exception as e:  # pragma: no cover
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
        category, date_str = f.stem.rsplit("_", 1)
        result.setdefault(category, []).append(date_str)
    return result