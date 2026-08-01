"""Monitoring and health check endpoints for the QA subsystem."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

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

    Returns a dictionary with the report contents. Raises HTTPException if the
    file exists but cannot be parsed or does not contain the expected structure.
    """
    if not HEALTH_REPORT_PATH.exists():
        return {"status": DEFAULT_HEALTH_STATUS, "message": DEFAULT_HEALTH_MESSAGE}
    try:
        raw = HEALTH_REPORT_PATH.read_text()
        report = json.loads(raw)
        if not isinstance(report, dict):
            raise ValueError("Health report is not a JSON object")
        # Ensure required keys exist; fill defaults if missing
        report.setdefault("status", DEFAULT_HEALTH_STATUS)
        report.setdefault("message", DEFAULT_HEALTH_MESSAGE)
        return report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=HEALTH_REPORT_CORRUPTED_DETAIL) from exc


def _read_fix_log(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read the fix log file and return the most recent *limit* entries.

    The log is stored as newline‑delimited JSON. Empty or missing files result in
    an empty list. Any parsing error raises an HTTPException.

    Args:
        limit: Maximum number of entries to return. If None or non‑positive,
               defaults to DEFAULT_FIX_LOG_LIMIT. Excessively large values are
               capped to avoid excessive memory usage.

    Returns:
        A list of dictionaries representing the most recent log entries,
        ordered from oldest to newest (newest last).
    """
    effective_limit = (
        DEFAULT_FIX_LOG_LIMIT if limit is None or limit <= 0 else limit
    )
    # Cap the limit to a reasonable maximum to prevent OOM
    MAX_LIMIT = 10_000
    if effective_limit > MAX_LIMIT:
        effective_limit = MAX_LIMIT

    if not FIX_LOG_PATH.exists():
        return []
    try:
        raw_text = FIX_LOG_PATH.read_text().strip()
        if not raw_text:
            return []
        lines = raw_text.splitlines()
        recent_lines = lines[-effective_limit:]
        entries: List[Dict[str, Any]] = []
        for line in recent_lines:
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    entries.append(entry)
                # Skip non‑dict entries silently
            except json.JSONDecodeError:
                # Skip malformed lines; continue processing others
                continue
        return entries
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=FIX_LOG_READ_ERROR_DETAIL.format(exc)
        ) from exc


@router.get("/health")
async def get_health_report():
    """Public health status (no auth required).

    Returns the most recent QA health report written by the QAMonitor background
    task, or a placeholder if the monitor has not yet completed its first cycle.
    """
    return _load_health_report()


@router.get("/fixes")
async def get_fix_log(
    limit: Optional[int] = DEFAULT_FIX_LOG_LIMIT,
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