"""Monitoring and health check endpoints for the QA subsystem."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator

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


class HealthReport(BaseModel):
    """Schema representing the QA subsystem health report."""

    status: str = Field(
        ...,
        description="Current health status of the QA subsystem.",
        example="healthy",
    )
    message: str = Field(
        ...,
        description="Human‑readable description of the health status.",
        example="All checks passed.",
    )
    details: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional additional details about the health check.",
        example={"latency_ms": 12},
    )

    @validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"healthy", "degraded", "unhealthy", "unknown"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return v

    class Config:
        extra = "allow"


class FixLogEntry(BaseModel):
    """Schema for a single auto‑fix log entry."""

    timestamp: str = Field(
        ...,
        description="ISO‑8601 timestamp when the fix was applied.",
        example="2026-07-25T12:34:56Z",
    )
    fix_type: str = Field(
        ...,
        description="Identifier of the auto‑fix that was attempted.",
        example="dependency_update",
    )
    result: str = Field(
        ...,
        description="Result of the fix attempt.",
        example="applied",
    )
    details: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional information about the fix.",
        example={"package": "numpy", "old_version": "1.22", "new_version": "1.23"},
    )

    @validator("timestamp")
    def validate_timestamp(cls, v: str) -> str:
        # Basic ISO‑8601 check; more thorough validation can be added if needed.
        if not isinstance(v, str) or "T" not in v:
            raise ValueError("timestamp must be an ISO‑8601 string")
        return v

    class Config:
        extra = "allow"


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
        raise HTTPException(status_code=500, detail=HEALTH_REPORT_CORRUPTED_DETAIL) from exc


def _read_fix_log(limit: int) -> List[FixLogEntry]:
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
        entries = [json.loads(line) for line in recent_lines]
        return [FixLogEntry(**entry) for entry in entries]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=FIX_LOG_READ_ERROR_DETAIL.format(exc)) from exc


@router.get("/health", response_model=HealthReport)
async def get_health_report():
    """Public health status (no auth required).

    Returns the most recent QA health report written by the QAMonitor background
    task, or a placeholder if the monitor has not yet completed its first cycle.
    """
    return _load_health_report()


@router.get("/fixes", response_model=List[FixLogEntry])
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