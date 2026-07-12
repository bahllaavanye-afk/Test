"""
Pipeline status API — reads pipeline_runs.json written by GitHub Actions scripts.
No database needed: the JSON file is the source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_STATE_FILE = Path(__file__).resolve().parents[5] / "pipeline_runs.json"

PIPELINE_DEFS = {
    "ml_experiments": {
        "label": "ML Experiments",
        "stages": [
            {"name": "data_fetch", "label": "Data Fetch", "channel": "#squad-data"},
            {"name": "cache_check", "label": "Cache Check", "channel": "#squad-data"},
            {"name": "feature_engineering", "label": "Feature Engineering", "channel": "#alpha-research"},
            {"name": "backtesting", "label": "Backtesting", "channel": "#ml-experiments"},
            {"name": "evaluation", "label": "Evaluation", "channel": "#ml-experiments"},
            {"name": "slack_report", "label": "Slack Report", "channel": "#ml-experiments"},
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
            {"name": "agent_posts", "label": "Slack Posts", "channel": "#engineering"},
        ],
    },
}


def _load_runs(limit: int = 50) -> List[Dict]:
    """Load the most recent pipeline runs from the JSON state file.

    Args:
        limit: Maximum number of runs to return. Non‑positive or non‑int values result in an empty list.

    Returns:
        A list of run dictionaries sorted by ``started_at`` descending, limited to ``limit`` items.
    """
    if not isinstance(limit, int) or limit <= 0:
        return []

    if not _STATE_FILE.exists():
        return []

    try:
        raw_text = _STATE_FILE.read_text()
        data = json.loads(raw_text)
        if not isinstance(data, list):
            return []
        # Ensure each element is a dict to avoid key errors during sorting
        valid_items = [item for item in data if isinstance(item, dict)]
        sorted_items = sorted(
            valid_items,
            key=lambda r: r.get("started_at", ""),
            reverse=True,
        )
        return sorted_items[:limit]
    except Exception:
        return []


def _enrich_run(run: Optional[Dict]) -> Dict:
    """Add stage definitions so the frontend knows the expected stage order.

    Handles ``None`` or malformed inputs gracefully by returning an empty dict.

    Args:
        run: The raw run dictionary.

    Returns:
        The enriched run dictionary with ``stages`` aligned to definitions.
    """
    if not isinstance(run, dict):
        return {}

    pipeline = run.get("pipeline", "")
    defn = PIPELINE_DEFS.get(pipeline, {})
    stage_defs = defn.get("stages", [])
    stage_order = [s.get("name") for s in stage_defs if isinstance(s, dict)]

    # Copy to avoid mutating the original
    enriched = dict(run)

    raw_stages = enriched.get("stages")
    if not isinstance(raw_stages, list):
        raw_stages = []

    # Index actual stage results by name
    actual: Dict[str, Dict] = {}
    for s in raw_stages:
        if isinstance(s, dict) and isinstance(s.get("name"), str):
            actual[s["name"]] = s

    # Build merged list: definition order, with actual data filled in
    merged: List[Dict] = []
    for sdef in stage_defs:
        if not isinstance(sdef, dict):
            continue
        sname = sdef.get("name")
        if sname in actual:
            merged.append({**sdef, **actual[sname]})
        else:
            merged.append({**sdef, "status": "pending"})

    # Append any extra stages not in definition, preserving order
    for s in raw_stages:
        if isinstance(s, dict) and s.get("name") not in stage_order:
            merged.append(s)

    enriched["stages"] = merged
    enriched["pipeline_label"] = defn.get("label", pipeline)
    return enriched


@router.get("/status")
def pipeline_status(
    pipeline: Optional[str] = Query(None),
    desk: Optional[str] = Query(None),
    limit: int = Query(20, le=50),
):
    """Return recent pipeline runs, optionally filtered by pipeline name or desk."""
    if not isinstance(limit, int) or limit <= 0:
        return []

    # Load a buffer larger than limit to allow filtering before final slice
    runs = _load_runs(limit * 2)
    if pipeline:
        runs = [r for r in runs if r.get("pipeline") == pipeline]
    if desk:
        runs = [r for r in runs if r.get("desk") == desk]

    # Guard against off‑by‑one slicing when limit exceeds available items
    return [_enrich_run(r) for r in runs[:limit]]


@router.get("/status/latest")
def pipeline_status_latest():
    """Return the most recent run for each pipeline type."""
    runs = _load_runs(100)
    seen: set[str] = set()
    latest: List[Dict] = []
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
    if not run_id:
        raise HTTPException(status_code=400, detail="Run ID must be provided")
    for run in _load_runs(100):
        if isinstance(run, dict) and run.get("run_id") == run_id:
            return _enrich_run(run)
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")


@router.get("/definitions")
def pipeline_definitions():
    """Return static pipeline stage definitions for the frontend."""
    return PIPELINE_DEFS