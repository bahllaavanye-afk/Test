"""Monitoring and health check endpoints for the QA subsystem."""
from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.models.user import User

# Constants
ROUTER_PREFIX = "/monitoring"
ROUTER_TAGS = ["monitoring"]
HEALTH_REPORT_PATH = Path(__file__).parents[4] / "qa_health_report.json"
FIX_LOG_PATH = Path(__file__).parents[4] / "qa_fix_log.jsonl"
DEFAULT_HEALTH_STATUS = "unknown"
DEFAULT_HEALTH_MESSAGE = "QA monitor not yet run"
DEFAULT_FIX_LOG_LIMIT = 50
HEALTH_REPORT_CORRUPTED_DETAIL = "Health report corrupted"
FIX_LOG_READ_ERROR_DETAIL = "Could not read fix log: {}"
QA_CYCLE_STARTED_MESSAGE = "QA cycle started — poll /monitoring/health for results"

router = APIRouter(prefix=ROUTER_PREFIX, tags=ROUTER_TAGS)

# Simple in‑memory cache for the health report to avoid repeated disk reads.
_health_report_cache: Dict[str, Any] = {"mtime": None, "data": None}


def _load_health_report() -> Dict[str, Any]:
    """Load the health report JSON from disk with caching.

    Returns a dictionary with the report contents. Raises HTTPException if the
    file exists but cannot be parsed.
    """
    if not HEALTH_REPORT_PATH.exists():
        return {"status": DEFAULT_HEALTH_STATUS, "message": DEFAULT_HEALTH_MESSAGE}

    try:
        mtime = os.path.getmtime(HEALTH_REPORT_PATH)
        cache = _health_report_cache
        if cache["mtime"] == mtime and cache["data"] is not None:
            return cache["data"]

        data = json.loads(HEALTH_REPORT_PATH.read_text())
        cache["mtime"] = mtime
        cache["data"] = data
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=HEALTH_REPORT_CORRUPTED_DETAIL) from exc


def _read_fix_log(limit: int) -> List[Dict[str, Any]]:
    """Read the fix log file and return the most recent *limit* entries.

    The log is stored as newline‑delimited JSON. Empty or missing files result in
    an empty list. Any parsing error raises an HTTPException.
    """
    if limit <= 0:
        return []
    if not FIX_LOG_PATH.exists():
        return []
    try:
        with FIX_LOG_PATH.open("r", encoding="utf-8") as f:
            recent_lines = deque(f, maxlen=limit)
        if not recent_lines:
            return []
        return [json.loads(line.rstrip("\n")) for line in recent_lines]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=FIX_LOG_READ_ERROR_DETAIL.format(exc)) from exc


@router.get("/health")
async def get_health_report():
    """Public health status (no auth required).

    Returns the most recent QA health report written by the QAMonitor background
    task, or a placeholder if the monitor has not yet completed its first cycle.
    """
    return _load_health_report()


@router.get("/fixes")
async def get_fix_log(
    limit: int = DEFAULT_FIX_LOG_LIMIT,
    current_user: User = Depends(get_current_user),
):
    """Recent auto‑fixes applied by the QA monitor (requires auth).

    Returns the last *limit* entries from the fix log (newest last).
    """
    return _read_fix_log(limit)


@router.post("/run-now")
async def trigger_qa_cycle(
    current_user: User = Depends(get_current_user),
):
    """Trigger an immediate QA cycle in the background (requires auth).

    The cycle runs asynchronously; poll GET /monitoring/health to see the result.
    """
    from app.tasks.qa_monitor import run_one_cycle

    asyncio.create_task(run_one_cycle())
    return {"message": QA_CYCLE_STARTED_MESSAGE}