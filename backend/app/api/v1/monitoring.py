"""Monitoring and health check endpoints for the QA subsystem."""
from __future__ import annotations

import asyncio
import collections
import json
import os
import pathlib
from functools import lru_cache
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

HEALTH_REPORT_PATH = pathlib.Path(__file__).parents[4] / "qa_health_report.json"
FIX_LOG_PATH = pathlib.Path(__file__).parents[4] / "qa_fix_log.jsonl"


def _file_mtime(path: pathlib.Path) -> float:
    """Return the modification time of *path* or 0 if it does not exist."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=1)
def _load_health_report_cached(mtime: float) -> dict:
    """Load the health report JSON from disk.

    The cache key is the file modification time, ensuring the content is refreshed
    only when the underlying file changes.
    """
    try:
        return json.loads(HEALTH_REPORT_PATH.read_text())
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Health report corrupted") from exc


@router.get("/health")
async def get_health_report() -> dict:
    """Public health status (no auth required).

    Returns the most recent QA health report written by the QAMonitor background
    task, or a placeholder if the monitor has not yet completed its first cycle.
    """
    if HEALTH_REPORT_PATH.exists():
        mtime = _file_mtime(HEALTH_REPORT_PATH)
        return _load_health_report_cached(mtime)
    return {"status": "unknown", "message": "QA monitor not yet run"}


def _read_last_n_lines(path: pathlib.Path, n: int) -> List[dict]:
    """Efficiently read the last *n* JSON lines from *path*.

    Uses a deque to keep only the needed number of lines in memory.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            dq = collections.deque(maxlen=n)
            for line in f:
                stripped = line.strip()
                if stripped:
                    dq.append(stripped)
        return [json.loads(line) for line in dq]
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Could not read fix log: {exc}") from exc


@router.get("/fixes")
async def get_fix_log(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
) -> List[dict]:
    """Recent auto-fixes applied by the QA monitor (requires auth).

    Returns the last *limit* entries from the fix log (newest last).
    """
    if not FIX_LOG_PATH.exists():
        return []
    # Fast path for empty file
    if FIX_LOG_PATH.stat().st_size == 0:
        return []
    return _read_last_n_lines(FIX_LOG_PATH, limit)


@router.post("/run-now")
async def trigger_qa_cycle(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Trigger an immediate QA cycle in the background (requires auth).

    The cycle runs asynchronously; poll GET /monitoring/health to see the result.
    """
    from app.tasks.qa_monitor import run_one_cycle

    asyncio.create_task(run_one_cycle())
    return {"message": "QA cycle started — poll /monitoring/health for results"}