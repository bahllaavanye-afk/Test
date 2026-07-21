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
    """
    Locate ``pipeline_runs.json`` without a hard‑coded ancestor depth.

    The old implementation used ``parents[5]`` which assumed a fixed directory
    depth and crashed the app when the file hierarchy differed. This function
    searches upward from the current file for ``pipeline_runs.json``, respects a
    ``PIPELINE_STATE_FILE`` environment override, and falls back to a reasonable
    default location. It never raises an exception at import time.
    """
    override = os.environ.get("PIPELINE_STATE_FILE")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pipeline_runs.json"
        if candidate.exists():
            return candidate
    idx = min(4, len(here.parents) - 1)
    return here.parents[idx] / "pipeline_runs.json"


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


def _load_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Load recent pipeline runs from the state file.

    Args:
        limit: Maximum number of runs to return after sorting by ``started_at``.

    Returns:
        A list of run dictionaries, possibly empty if the file does not exist or
        cannot be parsed.
    """
    if not _STATE_FILE.exists():
        return []
    try:
        data = json.loads(_STATE_FILE.read_text())
        if not isinstance(data, list):
            return []
        return sorted(data, key=lambda r: r.get("started_at", ""), reverse=True)[:limit]
    except Exception:
        return []


def _enrich_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add stage definitions to a raw run record so the frontend knows the expected order.

    The function merges the static stage definitions with the actual stage results,
    inserting a ``status`` of ``pending`` for missing stages and preserving any
    extra stages that are not defined.

    Args:
        run: A raw pipeline run dictionary.

    Returns:
        The run dictionary enriched with ``stages`` in definition order and a
        ``pipeline_label`` field.
    """
    pipeline = run.get("pipeline", "")
    defn = PIPELINE_DEFS.get(pipeline, {})
    stage_order = [s["name"] for s in defn.get("stages", [])]
    run = dict(run)

    # Index actual stage results by name
    actual: Dict[str, Dict[str, Any]] = {s["name"]: s for s in run.get("stages", [])}

    # Build merged list: definition order, with actual data filled in
    merged: List[Dict[str, Any]] = []
    for sdef in defn.get("stages", []):
        sname = sdef["name"]
        if sname in actual:
            merged.append({**sdef, **actual[sname]})
        else:
            merged.append({**sdef, "status": "pending"})

    # Append any extra stages not in definition
    for s in run.get("stages", []):
        if s["name"] not in stage_order:
            merged.append(s)

    run["stages"] = merged
    run["pipeline_label"] = defn.get("label", pipeline)
    return run


@router.get("/status")
def pipeline_status(
    pipeline: Optional[str] = Query(None),
    desk: Optional[str] = Query(None),
    limit: int = Query(20, le=50),
) -> List[Dict[str, Any]]:
    """
    Return recent pipeline runs, optionally filtered by pipeline name or desk.

    Args:
        pipeline: Filter runs to a specific pipeline identifier.
        desk: Filter runs to a specific desk identifier.
        limit: Maximum number of runs to return (capped at 50).

    Returns:
        A list of enriched pipeline run dictionaries.
    """
    runs = _load_runs(limit * 2)
    if pipeline:
        runs = [r for r in runs if r.get("pipeline") == pipeline]
    if desk:
        runs = [r for r in runs if r.get("desk") == desk]
    return [_enrich_run(r) for r in runs[:limit]]


@router.get("/status/latest")
def pipeline_status_latest() -> List[Dict[str, Any]]:
    """
    Return the most recent run for each distinct pipeline/desk combination.

    The function scans a larger window of recent runs and picks the first
    occurrence of each unique ``pipeline:desk`` key.

    Returns:
        A list of enriched run dictionaries, one per unique pipeline type.
    """
    runs = _load_runs(100)
    seen: set[str] = set()
    latest: List[Dict[str, Any]] = []
    for run in runs:
        key = f"{run.get('pipeline')}:{run.get('desk', '')}"
        if key not in seen:
            seen.add(key)
            latest.append(_enrich_run(run))
    return latest


@router.get("/status/{run_id}")
def pipeline_run_detail(run_id: str) -> Dict[str, Any]:
    """
    Return full detail for a specific pipeline run.

    Args:
        run_id: The unique identifier of the pipeline run to retrieve.

    Raises:
        HTTPException: If the run with the given ID does not exist.

    Returns:
        An enriched pipeline run dictionary.
    """
    for run in _load_runs(100):
        if run.get("run_id") == run_id:
            return _enrich_run(run)
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")


@router.get("/definitions")
def pipeline_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Return static pipeline stage definitions for the frontend.

    Returns:
        A dictionary mapping pipeline identifiers to their label and stage list.
    """
    return PIPELINE_DEFS