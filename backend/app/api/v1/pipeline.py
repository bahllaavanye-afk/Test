"""
Pipeline status API — reads pipeline_runs.json written by GitHub Actions scripts.
No database needed: the JSON file is the source of truth.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query

# Constants
PIPELINE_STATE_ENV = "PIPELINE_STATE_FILE"
PIPELINE_STATE_FILENAME = "pipeline_runs.json"
FALLBACK_PARENT_DEPTH = 4

DEFAULT_LOAD_LIMIT = 50
DEFAULT_QUERY_LIMIT = 20
MAX_QUERY_LIMIT = 50
DEFAULT_MULTIPLIER = 2
DEFAULT_LATEST_LIMIT = 100

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
    override = os.environ.get(PIPELINE_STATE_ENV)
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / PIPELINE_STATE_FILENAME
        if candidate.exists():
            return candidate
    # Fallback to a reasonable ancestor depth without risking IndexError
    idx = min(FALLBACK_PARENT_DEPTH, max(len(here.parents) - 1, 0))
    return here.parents[idx] / PIPELINE_STATE_FILENAME


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


def _load_runs(limit: int = DEFAULT_LOAD_LIMIT) -> List[Dict[str, Any]]:
    """Load recent pipeline runs from the JSON state file.

    Handles missing files, malformed JSON, and ensures the limit is a
    positive integer. Returns an empty list on any error or when limit <= 0.
    """
    if limit <= 0:
        return []
    if not _STATE_FILE.exists():
        return []
    try:
        raw_text = _STATE_FILE.read_text()
        data = json.loads(raw_text)
        if not isinstance(data, list):
            return []
        # Sort by start time descending; missing keys default to empty string.
        sorted_data = sorted(data, key=lambda r: r.get("started_at", ""), reverse=True)
        return sorted_data[:limit]
    except Exception:
        return []


def _enrich_run(run: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Add stage definitions so the frontend knows the expected stage order.

    Gracefully handles None or malformed run dictionaries and ensures that
    missing or empty stage collections do not raise errors.
    """
    if not isinstance(run, dict):
        return {}
    pipeline = run.get("pipeline", "")
    defn = PIPELINE_DEFS.get(pipeline, {})
    stage_defs = defn.get("stages", [])
    stage_order = [s.get("name") for s in stage_defs if isinstance(s, dict)]

    # Ensure stages is a list; fallback to empty list if malformed.
    raw_stages = run.get("stages", [])
    if not isinstance(raw_stages, list):
        raw_stages = []

    # Index actual stage results by name
    actual: Dict[str, Dict[str, Any]] = {
        s.get("name"): s for s in raw_stages if isinstance(s, dict) and "name" in s
    }

    # Build merged list: definition order, with actual data filled in
    merged: List[Dict[str, Any]] = []
    for sdef in stage_defs:
        if not isinstance(sdef, dict):
            continue
        sname = sdef.get("name")
        if sname in actual:
            merged.append({**sdef, **actual[sname]})
        else:
            merged.append({**sdef, "status": "pending"})

    # Append any extra stages not in definition, preserving order from raw_stages
    for s in raw_stages:
        if not isinstance(s, dict) or "name" not in s:
            continue
        if s["name"] not in stage_order:
            merged.append(s)

    enriched = dict(run)  # shallow copy
    enriched["stages"] = merged
    enriched["pipeline_label"] = defn.get("label", pipeline)
    return enriched


@router.get("/status")
def pipeline_status(
    pipeline: Optional[str] = Query(None),
    desk: Optional[str] = Query(None),
    limit: int = Query(DEFAULT_QUERY_LIMIT, le=MAX_QUERY_LIMIT),
):
    """Return recent pipeline runs, optionally filtered by pipeline name or desk."""
    # Guard against non‑positive limits
    if limit <= 0:
        return []
    runs = _load_runs(limit * DEFAULT_MULTIPLIER)
    if pipeline:
        runs = [r for r in runs if r.get("pipeline") == pipeline]
    if desk:
        runs = [r for r in runs if r.get("desk") == desk]
    # Slice safely even if runs is shorter than limit
    return [_enrich_run(r) for r in runs[:limit]]


@router.get("/status/latest")
def pipeline_status_latest():
    """Return the most recent run for each pipeline type."""
    runs = _load_runs(DEFAULT_LATEST_LIMIT)
    seen: set[str] = set()
    latest: List[Dict[str, Any]] = []
    for run in runs:
        key = f"{run.get('pipeline')}:{run.get('desk', '')}"
        if key not in seen:
            seen.add(key)
            latest.append(_enrich_run(run))
    return latest


@router.get("/status/{run_id}")
def pipeline_run_detail(run_id: str):
    """Return full detail for a specific pipeline run."""
    if not run_id:
        raise HTTPException(status_code=404, detail="Run identifier missing")
    for run in _load_runs(DEFAULT_LATEST_LIMIT):
        if run.get("run_id") == run_id:
            return _enrich_run(run)
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")


@router.get("/definitions")
def pipeline_definitions():
    """Return static pipeline stage definitions for the frontend."""
    return PIPELINE_DEFS