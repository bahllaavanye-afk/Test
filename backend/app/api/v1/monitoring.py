"""Monitoring and health check endpoints for the QA subsystem."""
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

# Endpoint paths
ENDPOINT_HEALTH = "/health"
ENDPOINT_FIXES = "/fixes"
ENDPOINT_RUN_NOW = "/run-now"

# Response keys
RESPONSE_MESSAGE_KEY = "message"

# HTTP status codes
HTTP_STATUS_INTERNAL_ERROR = 500

router = APIRouter(prefix=ROUTER_PREFIX, tags=ROUTER_TAGS)


def _load_health_report() -> Dict[str, Any]:
    """Load the health report JSON from disk.

    Returns a dictionary with the report contents. Raises HTTPException if the
    file exists but cannot be parsed.
    """
    if not HEALTH_REPORT_PATH.exists():
        return {"status": DEFAULT_HEALTH_STATUS, "message": DEFAULT_HEALTH_MESSAGE}
    try:
        return json.loads(HEALTH_REPORT_PATH.read_text())
    except Exception as exc:
        raise HTTPException(
            status_code=HTTP_STATUS_INTERNAL_ERROR,
            detail=HEALTH_REPORT_CORRUPTED_DETAIL,
        ) from exc


def _read_fix_log(limit: int) -> List[Dict[str, Any]]:
    """Read the fix log file and return the most recent *limit* entries.

    The log is stored as newline‑delimited JSON. Empty or missing files result in
    an empty list. Any parsing error raises an HTTPException.
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
        raise HTTPException(
            status_code=HTTP_STATUS_INTERNAL_ERROR,
            detail=FIX_LOG_READ_ERROR_DETAIL.format(exc),
        ) from exc


@router.get(ENDPOINT_HEALTH)
async def get_health_report():
    """Public health status (no auth required).

    Returns the most recent QA health report written by the QAMonitor background
    task, or a placeholder if the monitor has not yet completed its first cycle.
    """
    return _load_health_report()


@router.get(ENDPOINT_FIXES)
async def get_fix_log(
    limit: int = DEFAULT_FIX_LOG_LIMIT,
    current_user: User = Depends(get_current_user),
):
    """Recent auto‑fixes applied by the QA monitor (requires auth).

    Returns the last *limit* entries from the fix log (newest last).
    """
    return _read_fix_log(limit)


@router.post(ENDPOINT_RUN_NOW)
async def trigger_qa_cycle(
    current_user: User = Depends(get_current_user),
):
    """Trigger an immediate QA cycle in the background (requires auth).

    The cycle runs asynchronously; poll GET /monitoring/health to see the result.
    """
    from app.tasks.qa_monitor import run_one_cycle

    asyncio.create_task(run_one_cycle())
    return {RESPONSE_MESSAGE_KEY: QA_CYCLE_STARTED_MESSAGE}