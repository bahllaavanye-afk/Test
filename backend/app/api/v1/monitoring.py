"""Monitoring and health check endpoints for the QA subsystem."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user
from app.models.user import User

# Constants
ROUTER_PREFIX = "/monitoring"
ROUTER_TAGS = ["monitoring"]
HEALTH_REPORT_FILENAME = "qa_health_report.json"
FIX_LOG_FILENAME = "qa_fix_log.jsonl"
DEFAULT_FIX_LIMIT = 50
UNKNOWN_STATUS = "unknown"
UNKNOWN_MESSAGE = "QA monitor not yet run"
HEALTH_CORRUPTED_DETAIL = "Health report corrupted"
FIX_LOG_READ_ERROR_DETAIL = "Could not read fix log: {}"
RUN_NOW_MESSAGE = "QA cycle started — poll /monitoring/health for results"

router = APIRouter(prefix=ROUTER_PREFIX, tags=ROUTER_TAGS)

HEALTH_REPORT_PATH = Path(__file__).parents[4] / HEALTH_REPORT_FILENAME
FIX_LOG_PATH = Path(__file__).parents[4] / FIX_LOG_FILENAME


@router.get("/health")
async def get_health_report():
    """Public health status (no auth required).

    Returns the most recent QA health report written by the QAMonitor background
    task, or a placeholder if the monitor has not yet completed its first cycle.
    """
    if HEALTH_REPORT_PATH.exists():
        try:
            return json.loads(HEALTH_REPORT_PATH.read_text())
        except Exception:
            raise HTTPException(status_code=500, detail=HEALTH_CORRUPTED_DETAIL)
    return {"status": UNKNOWN_STATUS, "message": UNKNOWN_MESSAGE}


@router.get("/fixes")
async def get_fix_log(
    limit: int = DEFAULT_FIX_LIMIT,
    current_user: User = Depends(get_current_user),
):
    """Recent auto-fixes applied by the QA monitor (requires auth).

    Returns the last *limit* entries from the fix log (newest last).
    """
    if not FIX_LOG_PATH.exists():
        return []
    try:
        text = FIX_LOG_PATH.read_text().strip()
        if not text:
            return []
        lines = text.splitlines()
        return [json.loads(line) for line in lines[-limit:]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=FIX_LOG_READ_ERROR_DETAIL.format(e))


@router.post("/run-now")
async def trigger_qa_cycle(
    current_user: User = Depends(get_current_user),
):
    """Trigger an immediate QA cycle in the background (requires auth).

    The cycle runs asynchronously; poll GET /monitoring/health to see the result.
    """
    from app.tasks.qa_monitor import run_one_cycle

    asyncio.create_task(run_one_cycle())
    return {"message": RUN_NOW_MESSAGE}