"""Monitoring and health check endpoints for the QA subsystem.

Provides three endpoints:
- ``GET /monitoring/health`` – public health status of the QA monitor.
- ``GET /monitoring/fixes`` – recent auto‑fix entries (authenticated).
- ``POST /monitoring/run-now`` – trigger an immediate QA cycle (authenticated).

All endpoints return JSON‑serialisable objects; errors are reported via
``HTTPException`` with appropriate status codes.
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

    Returns:
        A dictionary with the report contents. If the file does not exist,
        returns a default placeholder. Raises ``HTTPException`` if the file
        exists but cannot be parsed.
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


def _read_fix_log_file() -> str:
    """Read the raw fix‑log file content.

    Returns:
        The stripped file content or an empty string if the file does not exist.
        Propagates parsing errors as ``HTTPException``.
    """
    if not FIX_LOG_PATH.exists():
        return ""
    try:
        return FIX_LOG_PATH.read_text().strip()
    except Exception as exc:
        raise HTTPException(
            status_code=HTTP_STATUS_INTERNAL_ERROR,
            detail=FIX_LOG_READ_ERROR_DETAIL.format(exc),
        ) from exc


def _parse_fix_log_lines(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse newline‑delimited JSON lines into a list of dictionaries.

    Args:
        lines: A list of JSON strings, each representing a log entry.

    Returns:
        A list of dictionaries parsed from the provided lines.

    Raises:
        HTTPException: If any line cannot be parsed as JSON.
    """
    try:
        return [json.loads(line) for line in lines]
    except Exception as exc:
        raise HTTPException(
            status_code=HTTP_STATUS_INTERNAL_ERROR,
            detail=FIX_LOG_READ_ERROR_DETAIL.format(exc),
        ) from exc


def _select_recent_lines(lines: List[str], limit: int) -> List[str]:
    """Select the most recent *limit* lines from the log.

    Args:
        lines: All log lines in chronological order.
        limit: Maximum number of lines to return; ``0`` returns an empty list.

    Returns:
        The last ``limit`` lines from ``lines``.
    """
    return lines[-limit:] if limit > 0 else []


def _read_fix_log(limit: int) -> List[Dict[str, Any]]:
    """Read the fix log and return the most recent *limit* entries.

    The log is stored as newline‑delimited JSON. Empty or missing files result in
    an empty list. Any parsing error raises an ``HTTPException``.

    Args:
        limit: Number of recent entries to return.

    Returns:
        A list of parsed log entry dictionaries.
    """
    raw_text = _read_fix_log_file()
    if not raw_text:
        return []
    lines = raw_text.splitlines()
    recent = _select_recent_lines(lines, limit)
    return _parse_fix_log_lines(recent)


@router.get(ENDPOINT_HEALTH)
async def get_health_report() -> Dict[str, Any]:
    """Public health status (no auth required).

    Returns the most recent QA health report written by the QAMonitor background
    task, or a placeholder if the monitor has not yet completed its first cycle.
    """
    return _load_health_report()


@router.get(ENDPOINT_FIXES)
async def get_fix_log(
    limit: int = DEFAULT_FIX_LOG_LIMIT,
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Recent auto‑fixes applied by the QA monitor (requires auth).

    Returns the last *limit* entries from the fix log (newest last).
    """
    return _read_fix_log(limit)


@router.post(ENDPOINT_RUN_NOW)
async def trigger_qa_cycle(
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """Trigger an immediate QA cycle in the background (requires auth).

    The cycle runs asynchronously; poll ``GET /monitoring/health`` to see the result.
    """
    from app.tasks.qa_monitor import run_one_cycle

    asyncio.create_task(run_one_cycle())
    return {RESPONSE_MESSAGE_KEY: QA_CYCLE_STARTED_MESSAGE}