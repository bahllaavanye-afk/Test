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
        raise HTTPException(status_code=500, detail=HEALTH_REPORT_CORRUPTED_DETAIL) from exc


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


# ---------------------------------------------------------------------------
# Unit tests for edge‑case behavior
# ---------------------------------------------------------------------------
import pytest
import tempfile


def _write_temp_file(path: Path, content: str) -> None:
    """Helper to write *content* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def temp_paths(monkeypatch):
    """Create isolated temporary health report and fix‑log files."""
    with tempfile.TemporaryDirectory() as td:
        health_path = Path(td) / "qa_health_report.json"
        fix_path = Path(td) / "qa_fix_log.jsonl"

        # Patch the module‑level constants to point at the temporary files
        monkeypatch.setattr(
            "backend.app.api.v1.monitoring.HEALTH_REPORT_PATH", health_path, raising=False
        )
        monkeypatch.setattr(
            "backend.app.api.v1.monitoring.FIX_LOG_PATH", fix_path, raising=False
        )
        yield health_path, fix_path


def test_load_health_report_missing_file(temp_paths):
    """When the health report file does not exist, defaults are returned."""
    health_path, _ = temp_paths
    # Ensure the file is absent
    if health_path.exists():
        health_path.unlink()
    result = _load_health_report()
    assert result["status"] == DEFAULT_HEALTH_STATUS
    assert result["message"] == DEFAULT_HEALTH_MESSAGE


def test_load_health_report_corrupted_json(temp_paths):
    """A malformed JSON file should raise an HTTPException with the correct detail."""
    health_path, _ = temp_paths
    _write_temp_file(health_path, "{invalid_json: true")  # deliberately corrupted
    with pytest.raises(HTTPException) as exc_info:
        _load_health_report()
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == HEALTH_REPORT_CORRUPTED_DETAIL


def test_read_fix_log_boundary_conditions(temp_paths):
    """Verify behavior for limit=0, limit larger than entries, and empty file."""
    _, fix_path = temp_paths

    # Empty file should yield empty list irrespective of limit
    _write_temp_file(fix_path, "")
    assert _read_fix_log(limit=0) == []
    assert _read_fix_log(limit=10) == []

    # Populate with three well‑formed JSON lines
    lines = [
        json.dumps({"id": 1, "action": "fix_a"}),
        json.dumps({"id": 2, "action": "fix_b"}),
        json.dumps({"id": 3, "action": "fix_c"}),
    ]
    _write_temp_file(fix_path, "\n".join(lines))

    # limit=0 returns empty list
    assert _read_fix_log(limit=0) == []

    # limit exceeds number of entries returns all entries preserving order
    all_entries = _read_fix_log(limit=10)
    assert len(all_entries) == 3
    assert all_entries == [json.loads(l) for l in lines]

    # limit=2 returns the last two entries
    recent_two = _read_fix_log(limit=2)
    assert recent_two == [json.loads(lines[1]), json.loads(lines[2])]


def test_read_fix_log_parsing_error(temp_paths):
    """If any line cannot be parsed as JSON, an HTTPException should be raised."""
    _, fix_path = temp_paths
    malformed = "not a json\n" + json.dumps({"valid": True})
    _write_temp_file(fix_path, malformed)
    with pytest.raises(HTTPException) as exc_info:
        _read_fix_log(limit=5)
    assert exc_info.value.status_code == 500
    assert "Could not read fix log" in exc_info.value.detail