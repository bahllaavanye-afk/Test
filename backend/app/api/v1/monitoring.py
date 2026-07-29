"""Monitoring and health check endpoints for the QA subsystem.

Provides public and authenticated endpoints to retrieve the QA health report,
access recent auto‑fix logs, and trigger an immediate QA monitoring cycle.
"""

from __future__ import annotations

import asyncio
import json
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


def _load_health_report() -> Dict[str, Any]:
    """Load the health report JSON from disk.

    Returns:
        A dictionary containing the health report fields. If the file does not
        exist, returns a placeholder with default status and message.

    Raises:
        HTTPException: If the file exists but cannot be parsed as JSON.
    """
    if not HEALTH_REPORT_PATH.exists():
        return {"status": DEFAULT_HEALTH_STATUS, "message": DEFAULT_HEALTH_MESSAGE}
    try:
        return json.loads(HEALTH_REPORT_PATH.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=HEALTH_REPORT_CORRUPTED_DETAIL) from exc


def _read_fix_log(limit: int) -> List[Dict[str, Any]]:
    """Read the fix log file and return the most recent *limit* entries.

    The log is stored as newline‑delimited JSON. Empty or missing files result in
    an empty list. Any parsing error raises an HTTPException.

    Args:
        limit: Maximum number of recent log entries to return.

    Returns:
        A list of dictionaries representing the most recent fix log entries.
    """
    if not FIX_LOG_PATH.exists():
        return []
    try:
        raw_text = FIX_LOG_PATH.read_text().strip()
        if not raw_text:
            return []
        lines = raw_text.splitlines()
        recent_lines = lines[-limit:]
        return [json.loads(line) for line in recent_lines]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=FIX_LOG_READ_ERROR_DETAIL.format(exc)) from exc


@router.get("/health")
async def get_health_report() -> Dict[str, Any]:
    """Public health status (no auth required).

    Returns the most recent QA health report written by the QAMonitor background
    task, or a placeholder if the monitor has not yet completed its first cycle.
    """
    return _load_health_report()


@router.get("/fixes")
async def get_fix_log(
    limit: int = DEFAULT_FIX_LOG_LIMIT,
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Recent auto‑fixes applied by the QA monitor (requires auth).

    Returns the last *limit* entries from the fix log (newest last).

    Args:
        limit: Maximum number of recent fix entries to retrieve.
        current_user: Authenticated user dependency; ensures the caller is authorized.

    Returns:
        List of fix log entries as dictionaries.
    """
    return _read_fix_log(limit)


@router.post("/run-now")
async def trigger_qa_cycle(
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """Trigger an immediate QA cycle in the background (requires auth).

    The cycle runs asynchronously; poll GET /monitoring/health to see the result.

    Args:
        current_user: Authenticated user dependency; ensures the caller is authorized.

    Returns:
        A message indicating that the QA cycle has been started.
    """
    from app.tasks.qa_monitor import run_one_cycle

    asyncio.create_task(run_one_cycle())
    return {"message": QA_CYCLE_STARTED_MESSAGE}