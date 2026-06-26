"""
Pipeline status API — reads pipeline_runs.json written by GitHub Actions scripts.
No database needed: the JSON file is the source of truth.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_STATE_FILE = Path(__file__).resolve().parents[5] / "pipeline_runs.json"
_logger = logging.getLogger(__name__)

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
    """Load recent pipeline runs from the JSON state file.

    Args:
        limit: Maximum number of runs to return. Non‑positive values return an empty list.

    Returns:
        A list of run dictionaries sorted by ``started_at`` descending.
    """
    if not isinstance(limit, int) or limit <= 0:
        _logger.debug("Invalid limit %s supplied to _load_runs; returning empty list.", limit)
        return []

    if not _STATE_FILE.exists():
        _logger.debug("State file %s does not exist.", _STATE_FILE)
        return []

    try:
        raw = _STATE_FILE.read_text()
        data = json.loads(raw)
        if not isinstance(data, list):
            _logger.warning("State file does not contain a list; got %s", type(data))
            return []
        sorted_data = sorted(data, key=lambda r: r.get("started_at", ""), reverse=True)
        return sorted_data[:limit]
    except Exception as exc:  # pragma: no cover
        _logger.exception("Failed to load pipeline runs: %s", exc)
        return []


def _enrich_run(run: Dict[str, Any] | None) -> Dict[str, Any]:
    """Add stage definitions so the frontend knows the expected stage order.

    Handles ``None`` or malformed inputs gracefully.

    Args:
        run: The raw run dictionary.

    Returns:
        An enriched run dictionary (may be empty if input is invalid).
    """
    if not isinstance(run, dict):
        _logger.debug("Enrich called with non‑dict run: %s", run)
        return {}

    pipeline = run.get("pipeline", "")
    defn = PIPELINE_DEFS.get(pipeline, {})
    stage_defs = defn.get("stages", [])
    stage_order = [s.get("name") for s in stage_defs if isinstance(s, dict)]

    # Ensure stages is a list; fallback to empty list if missing or malformed
    raw_stages = run.get("stages", [])
    if not isinstance(raw_stages, list):
        _logger.debug("Run %s stages field is not a list; resetting to empty.", run.get("run_id"))
        raw_stages = []

    # Index actual stage results by name, ignoring entries without a name
    actual: Dict[str, Dict[str, Any]] = {}
    for s in raw_stages:
        if isinstance(s, dict) and "name" in s:
            actual[s["name"]] = s

    merged: List[Dict[str, Any]] = []
    for sdef in stage_defs:
        if not isinstance(sdef, dict):
            continue
        sname = sdef.get("name")
        if not sname:
            continue
        if sname in actual:
            merged.append({**sdef, **actual[sname]})
        else:
            merged.append({**sdef, "status": "pending"})

    # Append any extra stages not in definition, preserving order
    for s in raw_stages:
        if isinstance(s, dict) and s.get("name") not in stage_order:
            merged.append(s)

    enriched = dict(run)  # shallow copy
    enriched["stages"] = merged
    enriched["pipeline_label"] = defn.get("label", pipeline)
    return enriched


@router.get("/status")
def pipeline_status(
    pipeline: str | None = Query(None),
    desk: str | None = Query(None),
    limit: int = Query(20, le=50),
):
    """Return recent pipeline runs, optionally filtered by pipeline name or desk."""
    # Guard against non‑positive limits; FastAPI enforces int but callers may pass 0
    effective_limit = limit if limit and limit > 0 else 20
    runs = _load_runs(effective_limit * 2)

    if pipeline:
        runs = [r for r in runs if r.get("pipeline") == pipeline]
    if desk:
        runs = [r for r in runs if r.get("desk") == desk]

    # Slice safely even if runs list is shorter than requested
    return [_enrich_run(r) for r in runs[:effective_limit]]


@router.get("/status/latest")
def pipeline_status_latest():
    """Return the most recent run for each pipeline type."""
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
def pipeline_run_detail(run_id: str):
    """Return full detail for a specific pipeline run."""
    for run in _load_runs(100):
        if run.get("run_id") == run_id:
            return _enrich_run(run)
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")


@router.get("/definitions")
def pipeline_definitions():
    """Return static pipeline stage definitions for the frontend."""
    return PIPELINE_DEFS