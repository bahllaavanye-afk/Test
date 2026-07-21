"""Monitoring and health check endpoints for the QA subsystem."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, validator

from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

HEALTH_REPORT_PATH = Path(__file__).parents[4] / "qa_health_report.json"
FIX_LOG_PATH = Path(__file__).parents[4] / "qa_fix_log.jsonl"


class HealthReport(BaseModel):
    """Schema representing the QA health report."""

    status: str = Field(
        ...,
        description="Overall health status of the QA subsystem.",
        example="healthy",
    )
    message: str = Field(
        ...,
        description="Human‑readable description of the current health state.",
        example="All systems operational.",
    )

    @validator("status")
    def validate_status(cls, v: str) -> str:
        allowed = {"healthy", "degraded", "unknown"}
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
        example="2026-07-20T14:32:10Z",
    )
    fix_type: str = Field(
        ...,
        description="Identifier of the fix type or rule that was triggered.",
        example="auto_patch_failure",
    )
    details: dict = Field(
        default_factory=dict,
        description="Arbitrary details describing the fix action.",
        example={"file": "src/module.py", "line": 42},
    )

    @validator("timestamp")
    def validate_timestamp(cls, v: str) -> str:
        # Simple validation to ensure ISO‑8601 format; more rigorous checks can be added.
        if not v or "T" not in v:
            raise ValueError("timestamp must be a valid ISO‑8601 string")
        return v


@router.get("/health", response_model=HealthReport)
async def get_health_report():
    """Public health status (no auth required).

    Returns the most recent QA health report written by the QAMonitor background
    task, or a placeholder if the monitor has not yet completed its first cycle.
    """
    if HEALTH_REPORT_PATH.exists():
        try:
            data = json.loads(HEALTH_REPORT_PATH.read_text())
            return HealthReport(**data)
        except Exception:
            raise HTTPException(status_code=500, detail="Health report corrupted")
    # Placeholder when no report is available yet
    return HealthReport(status="unknown", message="QA monitor not yet run")


@router.get("/fixes", response_model=List[FixLogEntry])
async def get_fix_log(
    limit: int = 50,
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
        recent = lines[-limit:]
        return [FixLogEntry(**json.loads(line)) for line in recent]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read fix log: {e}")


@router.post("/run-now")
async def trigger_qa_cycle(
    current_user: User = Depends(get_current_user),
):
    """Trigger an immediate QA cycle in the background (requires auth).

    The cycle runs asynchronously; poll GET /monitoring/health to see the result.
    """
    from app.tasks.qa_monitor import run_one_cycle

    asyncio.create_task(run_one_cycle())
    return {"message": "QA cycle started — poll /monitoring/health for results"}