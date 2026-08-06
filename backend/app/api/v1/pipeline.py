"""
Pipeline status API — reads pipeline_runs.json written by GitHub Actions scripts.
No database needed: the JSON file is the source of truth.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

# Constants
ENV_PIPELINE_STATE_FILE = "PIPELINE_STATE_FILE"
FILE_PIPELINE_STATE = "pipeline_runs.json"
FALLBACK_PARENT_DEPTH = 4

DEFAULT_LOAD_LIMIT = 50
DEFAULT_QUERY_LIMIT = 20
MAX_QUERY_LIMIT = 50
DEFAULT_MULTIPLIER = 2
DEFAULT_LATEST_LIMIT = 100

# JSON keys
KEY_PIPELINE = "pipeline"
KEY_DESK = "desk"
KEY_RUN_ID = "run_id"
KEY_STARTED_AT = "started_at"
KEY_STAGES = "stages"
KEY_STATUS = "status"
KEY_PENDING = "pending"
KEY_PIPELINE_LABEL = "pipeline_label"

# Error messages
ERR_RUN_NOT_FOUND = "Run {run_id!r} not found"

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _resolve_state_file() -> Path:
    """Locate pipeline_runs.json without a hardcoded ancestor depth.

    The old `parents[5]` assumed a fixed directory depth and crashed the ENTIRE
    app at import on Render (IndexError: 5) — there the file is /app/app/api/v1/
    pipeline.py, only 5 parents (0-4), so index 5 doesn't exist. This searches
    ancestors for the file (it's written to the repo root by GitHub Actions),
    honours a PIPELINE_STATE_FILE override, and falls back to a repo-root-ish
    default that simply may not exist (readers already handle absence). Never
    raises at import.
    """
    override = os.environ.get(ENV_PIPELINE_STATE_FILE)
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / FILE_PIPELINE_STATE
        if candidate.exists():
            return candidate
    idx = min(FALLBACK_PARENT_DEPTH, len(here.parents) - 1)
    return here.parents[idx] / FILE_PIPELINE_STATE


_STATE_FILE = _resolve_state_file()

PIPELINE_DEFS = {
    "ml_experiments": {
        "label": "ML Experiments",
        "stages": [
            {"name": "data_fetch",          "label": "Data Fetch",          "channel": "#squad-data"},
            {"name": "cache_check",         "label": "Cache Check",         "channel": "#squad-data"},
            {"name": "feature_engineering", "label": "Feature Engineering", "channel": "#alpha-research"},
            {"name": "backtesting",         "label": "Backtesting",         "channel": "#ml-experiments"},
            {"name": "evaluation",          "label": "Evaluation",          "channel": "#ml-experiments"},
            {"name": "employee_report",     "label": "Employee Report",     "channel": "#ml-experiments"},
            {"name": "commit_results",      "label": "Commit Results",      "channel": None},
        ],
    },
    "desk_trading": {
        "label": "Desk Trading",
        "stages": [
            {"name": "market_status",    "label": "Market Status",    "channel": None},
            {"name": "data_fetch",       "label": "Data Fetch",       "channel": "#squad-data"},
            {"name": "signal_generation","label": "Signal Generation","channel": None},
            {"name": "risk_check",       "label": "Risk Check",       "channel": "#risk-alerts"},
            {"name": "order_execution",  "label": "Order Execution",  "channel": None},
            {"name": "fill_tracking",    "label": "Fill Tracking",    "channel": None},
            {"name": "pnl_snapshot",     "label": "P&L Snapshot",     "channel": "#pnl-daily"},
        ],
    },
    "agent_team": {
        "label": "Agent Team",
        "stages": [
            {"name": "data_fetch",    "label": "Data Fetch",    "channel": None},
            {"name": "agent_dispatch","label": "Agent Dispatch","channel": None},
            {"name": "agent_posts",   "label": "Discord Posts", "channel": "#engineering"},
        ],
    },
}


def _load_runs(limit: int = DEFAULT_LOAD_LIMIT) -> list[dict]:
    if not _STATE_FILE.exists():
        return []
    try:
        data = json.loads(_STATE_FILE.read_text())
        if not isinstance(data, list):
            return []
        return sorted(data, key=lambda r: r.get(KEY_STARTED_AT, ""), reverse=True)[:limit]
    except Exception:
        return []


def _enrich_run(run: dict) -> dict:
    """Add stage definitions so the frontend knows the expected stage order."""
    pipeline = run.get(KEY_PIPELINE, "")
    defn = PIPELINE_DEFS.get(pipeline, {})
    stage_order = [s["name"] for s in defn.get("stages", [])]
    run = dict(run)

    # Index actual stage results by name
    actual: dict[str, dict] = {s["name"]: s for s in run.get(KEY_STAGES, [])}

    # Build merged list: definition order, with actual data filled in
    merged = []
    for sdef in defn.get("stages", []):
        sname = sdef["name"]
        if sname in actual:
            merged.append({**sdef, **actual[sname]})
        else:
            merged.append({**sdef, KEY_STATUS: KEY_PENDING})

    # Append any extra stages not in definition
    for s in run.get(KEY_STAGES, []):
        if s["name"] not in stage_order:
            merged.append(s)

    run[KEY_STAGES] = merged
    run[KEY_PIPELINE_LABEL] = defn.get("label", pipeline)
    return run


@router.get("/status")
def pipeline_status(
    pipeline: Optional[str] = Query(None),
    desk: Optional[str] = Query(None),
    limit: int = Query(DEFAULT_QUERY_LIMIT, le=MAX_QUERY_LIMIT),
):
    """Return recent pipeline runs, optionally filtered by pipeline name or desk."""
    runs = _load_runs(limit * DEFAULT_MULTIPLIER)
    if pipeline:
        runs = [r for r in runs if r.get(KEY_PIPELINE) == pipeline]
    if desk:
        runs = [r for r in runs if r.get(KEY_DESK) == desk]
    return [_enrich_run(r) for r in runs[:limit]]


@router.get("/status/latest")
def pipeline_status_latest():
    """Return the most recent run for each pipeline type."""
    runs = _load_runs(DEFAULT_LATEST_LIMIT)
    seen: set[str] = set()
    latest: list[dict] = []
    for run in runs:
        key = f"{run.get(KEY_PIPELINE)}:{run.get(KEY_DESK, '')}"
        if key not in seen:
            seen.add(key)
            latest.append(_enrich_run(run))
    return latest


@router.get("/status/{run_id}")
def pipeline_run_detail(run_id: str):
    """Return full detail for a specific pipeline run."""
    for run in _load_runs(DEFAULT_LATEST_LIMIT):
        if run.get(KEY_RUN_ID) == run_id:
            return _enrich_run(run)
    raise HTTPException(status_code=404, detail=ERR_RUN_NOT_FOUND.format(run_id=run_id))


@router.get("/definitions")
def pipeline_definitions():
    """Return static pipeline stage definitions for the frontend."""
    return PIPELINE_DEFS