"""
Pipeline status API — reads pipeline_runs.json written by GitHub Actions scripts.
No database needed: the JSON file is the source of truth.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

def _resolve_state_file() -> Path:
    """Locate ``pipeline_runs.json`` without a hard‑coded ancestor depth.

    The previous implementation used ``parents[5]`` which raised ``IndexError``
    when the directory depth was smaller than expected.  This function walks
    up the directory tree looking for the file, respects the ``PIPELINE_STATE_FILE``
    environment override, and falls back to a reasonable default without raising
    at import time.
    """
    override = os.environ.get("PIPELINE_STATE_FILE")
    if override:
        return Path(override)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pipeline_runs.json"
        if candidate.is_file():
            return candidate

    # Fallback: use a parent that is guaranteed to exist (at most 4 levels up)
    fallback_idx = min(4, len(here.parents) - 1)
    return here.parents[fallback_idx] / "pipeline_runs.json"


_STATE_FILE = _resolve_state_file()

PIPELINE_DEFS: Dict[str, Dict[str, Any]] = {
    "ml_experiments": {
        "label": "ML Experiments",
        "stages": [
            {"name": "data_fetch", "label": "Data Fetch", "channel": "#squad-data"},
            {"name": "cache_check", "label": "Cache Check", "channel": "#squad-data"},
            {"name": "feature_engineering", "label": "Feature Engineering", "channel": "#alpha-research"},
            {"name": "backtesting", "label": "Backtesting", "channel": "#ml-experiments"},
            {"name": "evaluation", "label": "Evaluation", "channel": "#ml-experiments"},
            {"name": "employee_report", "label": "Employee Report", "channel": "#ml-experiments"},
            {"name": "commit_results", "label": "Commit Results", "channel": None},
        ],
    },
    "desk_trading": {
        "label": "Desk Trading",
        "stages": [
            {"name": "market_status", "label": "Market Status", "channel": None},
            {"name": "data_fetch", "label": "Data Fetch", "channel": "#squad-data"},
            {"name": "signal_generation", "label": "Signal Generation", "channel": None},
            {"name": "risk_check", "label": "Risk Check", "channel": "#risk-alerts"},
            {"name": "order_execution", "label": "Order Execution", "channel": None},
            {"name": "fill_tracking", "label": "Fill Tracking", "channel": None},
            {"name": "pnl_snapshot", "label": "P&L Snapshot", "channel": "#pnl-daily"},
        ],
    },
    "agent_team": {
        "label": "Agent Team",
        "stages": [
            {"name": "data_fetch", "label": "Data Fetch", "channel": None},
            {"name": "agent_dispatch", "label": "Agent Dispatch", "channel": None},
            {"name": "agent_posts", "label": "Discord Posts", "channel": "#engineering"},
        ],
    },
}

def _load_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """Load recent pipeline runs from the state file.

    Returns an empty list when the file is missing, malformed, or when ``limit``
    is non‑positive.  The function is defensive against ``None`` or unexpected
    data structures.
    """
    if limit <= 0:
        return []

    if not _STATE_FILE.is_file():
        return []

    try:
        raw_text = _STATE_FILE.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        if not isinstance(data, list):
            return []
        # Sort by ``started_at`` descending; missing keys default to empty string.
        sorted_data = sorted(data, key=lambda r: r.get("started_at", ""), reverse=True)
        return sorted_data[:limit]
    except Exception:
        return []

def _enrich_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Add stage definitions so the frontend knows the expected stage order.

    Handles ``None`` or malformed ``run`` dictionaries gracefully, and copes
    with missing or empty ``stages`` collections.
    """
    if not isinstance(run, dict):
        return {}

    pipeline = run.get("pipeline", "")
    defn = PIPELINE_DEFS.get(pipeline, {})
    stage_defs = defn.get("stages", [])
    if not isinstance(stage_defs, list):
        stage_defs = []

    stage_order = [s.get("name") for s in stage_defs if isinstance(s, dict) and "name" in s]

    # Ensure we have a list for actual stages
    actual_stages = run.get("stages", [])
    if not isinstance(actual_stages, list):
        actual_stages = []

    # Index actual stage results by name
    actual: Dict[str, Dict[str, Any]] = {}
    for s in actual_stages:
        if isinstance(s, dict):
            name = s.get("name")
            if isinstance(name, str):
                actual[name] = s

    # Build merged list: definition order, with actual data filled in
    merged: List[Dict[str, Any]] = []
    for sdef in stage_defs:
        if not isinstance(sdef, dict):
            continue
        sname = sdef.get("name")
        if not isinstance(sname, str):
            continue
        if sname in actual:
            merged.append({**sdef, **actual[sname]})
        else:
            merged.append({**sdef, "status": "pending"})

    # Append any extra stages not in definition
    for s in actual_stages:
        if not isinstance(s, dict):
            continue
        sname = s.get("name")
        if sname not in stage_order:
            merged.append(s)

    enriched = dict(run)  # shallow copy
    enriched["stages"] = merged
    enriched["pipeline_label"] = defn.get("label", pipeline)
    return enriched

@router.get("/status")
def pipeline_status(
    pipeline: Optional[str] = Query(None),
    desk: Optional[str] = Query(None),
    limit: int = Query(20, le=50),
):
    """Return recent pipeline runs, optionally filtered by pipeline name or desk.

    ``limit`` is capped at 50 by the query validator; we also guard against
    non‑positive values to avoid off‑by‑one slicing issues.
    """
    if limit <= 0:
        return []

    # Load a buffer larger than the requested limit to allow filtering before slicing.
    runs = _load_runs(limit * 2)
    if pipeline:
        runs = [r for r in runs if r.get("pipeline") == pipeline]
    if desk:
        runs = [r for r in runs if r.get("desk") == desk]
    # Slice to the exact limit after filtering.
    return [_enrich_run(r) for r in runs[:limit]]

@router.get("/status/latest")
def pipeline_status_latest():
    """Return the most recent run for each distinct ``pipeline:desk`` combination."""
    runs = _load_runs(100)
    seen: set[str] = set()
    latest: List[Dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        key = f"{run.get('pipeline')}:{run.get('desk', '')}"
        if key not in seen:
            seen.add(key)
            latest.append(_enrich_run(run))
    return latest

@router.get("/status/{run_id}")
def pipeline_run_detail(run_id: str):
    """Return full detail for a specific pipeline run."""
    for run in _load_runs(100):
        if isinstance(run, dict) and run.get("run_id") == run_id:
            return _enrich_run(run)
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

@router.get("/definitions")
def pipeline_definitions():
    """Return static pipeline stage definitions for the frontend."""
    return PIPELINE_DEFS