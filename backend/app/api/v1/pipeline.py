"""
Pipeline status API — reads pipeline_runs.json written by GitHub Actions scripts.
No database needed: the JSON file is the source of truth.
"""
from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

logger = logging.getLogger(__name__)

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


def _load_runs(limit: int = 50) -> List[dict]:
    """Load pipeline run entries from the JSON state file.

    Returns a list of run dictionaries limited to the most recent *limit* entries.
    Errors while reading or parsing the file are logged and result in an empty list.
    """
    if not _STATE_FILE.exists():
        logger.debug("Pipeline state file %s does not exist; returning empty list.", _STATE_FILE)
        return []
    try:
        raw_text = _STATE_FILE.read_text()
        data = json.loads(raw_text)
        if not isinstance(data, list):
            logger.warning(
                "Unexpected data format in %s: expected list, got %s.",
                _STATE_FILE,
                type(data).__name__,
            )
            return []
        sorted_data = sorted(data, key=lambda r: r.get("started_at", ""), reverse=True)
        return sorted_data[:limit]
    except FileNotFoundError as exc:
        logger.error("State file not found: %s", exc)
        return []
    except PermissionError as exc:
        logger.error("Permission error accessing state file %s: %s", _STATE_FILE, exc)
        return []
    except JSONDecodeError as exc:
        logger.error("JSON decode error in state file %s: %s", _STATE_FILE, exc)
        return []
    except Exception as exc:
        logger.exception("Unexpected error loading pipeline runs: %s", exc)
        return []


def _enrich_run(run: dict) -> dict:
    """Add stage definitions so the frontend knows the expected stage order."""
    pipeline = run.get("pipeline", "")
    defn = PIPELINE_DEFS.get(pipeline, {})
    stage_order = [s["name"] for s in defn.get("stages", [])]
    enriched = dict(run)

    # Index actual stage results by name
    actual: dict[str, dict] = {s["name"]: s for s in run.get("stages", []) if isinstance(s, dict)}

    # Build merged list: definition order, with actual data filled in
    merged: List[dict] = []
    for sdef in defn.get("stages", []):
        sname = sdef["name"]
        if sname in actual:
            merged.append({**sdef, **actual[sname]})
        else:
            merged.append({**sdef, "status": "pending"})

    # Append any extra stages not in definition
    for s in run.get("stages", []):
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
) -> List[dict]:
    """Return recent pipeline runs, optionally filtered by pipeline name or desk."""
    try:
        runs = _load_runs(limit * 2)
        if pipeline:
            runs = [r for r in runs if r.get("pipeline") == pipeline]
        if desk:
            runs = [r for r in runs if r.get("desk") == desk]
        return [_enrich_run(r) for r in runs[:limit]]
    except Exception as exc:
        logger.exception("Failed to retrieve pipeline status: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to retrieve pipeline status") from exc


@router.get("/status/latest")
def pipeline_status_latest() -> List[dict]:
    """Return the most recent run for each pipeline type."""
    try:
        runs = _load_runs(100)
        seen: set[str] = set()
        latest: List[dict] = []
        for run in runs:
            key = f"{run.get('pipeline')}:{run.get('desk', '')}"
            if key not in seen:
                seen.add(key)
                latest.append(_enrich_run(run))
        return latest
    except Exception as exc:
        logger.exception("Failed to retrieve latest pipeline statuses: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to retrieve latest pipeline statuses") from exc


@router.get("/status/{run_id}")
def pipeline_run_detail(run_id: str) -> dict:
    """Return full detail for a specific pipeline run."""
    try:
        for run in _load_runs(100):
            if run.get("run_id") == run_id:
                return _enrich_run(run)
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error retrieving pipeline run %s: %s", run_id, exc)
        raise HTTPException(status_code=500, detail="Unable to retrieve pipeline run details") from exc


@router.get("/definitions")
def pipeline_definitions() -> dict:
    """Return static pipeline stage definitions for the frontend."""
    try:
        return PIPELINE_DEFS
    except Exception as exc:
        logger.exception("Failed to return pipeline definitions: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to retrieve pipeline definitions") from exc