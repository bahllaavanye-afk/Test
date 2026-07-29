"""
Trade Archiver: writes every order, fill, and signal to JSON‑lines files for long‑term audit
and replay. Files rotate daily.
"""

from __future__ import annotations

import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from app.utils.logging import logger

# Base directory for archived JSON‑lines files.
ARCHIVE_DIR = Path(__file__).parents[3] / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Global asyncio lock to guarantee atomic writes.
_lock = asyncio.Lock()


def _today_file(category: str) -> Path:
    """
    Build the path for today's archive file for a given category.

    Args:
        category: Archive category (e.g., ``orders``, ``fills``).

    Returns:
        Path to the JSON‑lines file named ``{category}_YYYY‑MM‑DD.jsonl``.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ARCHIVE_DIR / f"{category}_{date_str}.jsonl"


def _sync_append(path: str, line: str) -> None:
    """
    Synchronously append a line to a file.

    This helper runs in a thread pool via ``run_in_executor`` to avoid blocking the
    event loop.

    Args:
        path: Filesystem path to the target file.
        line: Text line to append (including newline character).
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


async def archive_event(category: str, data: Dict[str, Any]) -> None:
    """
    Append a single JSON‑encoded event to today's archive file for the given category.

    The operation is protected by an asyncio lock to ensure atomic writes across
    concurrent callers.

    Args:
        category: One of ``orders``, ``fills``, ``signals``, ``decisions``, or ``risk``.
        data: Arbitrary payload that will be merged with a timestamp.

    Returns:
        None
    """
    record = {"ts": datetime.now(timezone.utc).isoformat(), **data}
    line = json.dumps(record, default=str) + "\n"
    file = _today_file(category)
    try:
        loop = asyncio.get_running_loop()
        async with _lock:
            await loop.run_in_executor(None, _sync_append, str(file), line)
    except Exception as e:
        logger.warning("Archive failed", category=category, error=str(e))


def replay(category: str, date_str: str | None = None, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Retrieve archived events for a specific category and date.

    Args:
        category: Archive category to read.
        date_str: Date in ``YYYY-MM-DD`` format. If ``None``, uses today's UTC date.
        limit: Maximum number of records to return.

    Returns:
        List of decoded JSON objects (each as a ``dict``). Returns an empty list if
        the file does not exist or cannot be parsed.
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
            except Exception:
                continue
    return out


def list_archives() -> dict[str, List[str]]:
    """
    List all available archive files grouped by category.

    Returns:
        Mapping from category name to a list of date strings (``YYYY-MM-DD``) for
        which an archive file exists, sorted chronologically.
    """
    result: dict[str, List[str]] = {}
    for f in sorted(ARCHIVE_DIR.glob("*.jsonl")):
        category, date_str = f.stem.rsplit("_", 1)
        result.setdefault(category, []).append(date_str)
    return result