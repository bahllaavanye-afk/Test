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
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ARCHIVE_DIR / f"{category}_{date_str}.jsonl"


async def archive_event(category: str, data: dict) -> None:
    """
    category: 'orders' | 'fills' | 'signals' | 'decisions' | 'risk'
    Appends a single JSON line to today's file. Atomic (lock-guarded).
    """
    record = {"ts": datetime.now(timezone.utc).isoformat(), **data}
    line = json.dumps(record, default=str) + "\n"
    file = _today_file(category)
    try:
        async with _lock:
            with open(file, "a", encoding="utf-8") as f:
                f.write(line)
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
