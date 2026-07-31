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

_ALLOWED_CATEGORIES = {"orders", "fills", "signals", "decisions", "risk"}


def _today_file(category: str) -> Path:
    """Return the Path for today's archive file of the given category."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ARCHIVE_DIR / f"{category}_{date_str}.jsonl"


def _sync_append(path: Path, line: str) -> None:
    """Synchronously append a line to a file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


async def archive_event(category: str, data: dict) -> None:
    """
    Append a single JSON line to today's file for the given category.
    The operation is guarded by an async lock to ensure atomicity.
    """
    if category not in _allowed_categories:
        logger.warning("Attempted to archive unknown category", category=category)
        return

    record = {"ts": datetime.now(timezone.utc).isoformat(), **data}
    line = json.dumps(record, default=str) + "\n"
    file_path = _today_file(category)

    try:
        loop = asyncio.get_running_loop()
        async with _lock:
            await loop.run_in_executor(None, _sync_append, file_path, line)
    except Exception as e:  # pragma: no cover
        logger.warning("Archive failed", category=category, error=str(e))


def replay(category: str, date_str: str | None = None, limit: int = 1000) -> list[dict]:
    """
    Read back archived events for a category and date (YYYY-MM-DD).

    Returns up to ``limit`` records; an empty list is returned if the file does not exist.
    """
    if category not in _allowed_categories:
        logger.warning("Attempted to replay unknown category", category=category)
        return []

    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    file_path = ARCHIVE_DIR / f"{category}_{date_str}.jsonl"
    if not file_path.exists():
        return []

    out: list[dict] = []
    with file_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
                if len(out) >= limit:
                    break
            except Exception:
                continue
    return out


def list_archives() -> dict[str, list[str]]:
    """Return a mapping of category to sorted list of available dates."""
    result: dict[str, list[str]] = {}
    for file_path in sorted(ARCHIVE_DIR.glob("*.jsonl")):
        try:
            category, date_str = file_path.stem.rsplit("_", 1)
        except ValueError:
            continue
        result.setdefault(category, []).append(date_str)
    return result