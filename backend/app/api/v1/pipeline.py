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

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

def _resolve_state_file() -> Path:
    """Locate ``pipeline_runs.json`` without assuming a fixed directory depth.

    The previous implementation used ``parents[5]`` which fails when the
    repository layout differs (e.g., on Render). This function searches
    upward from the current file for the JSON file, respects a
    ``PIPELINE_STATE_FILE`` environment override, and falls back to a
    reasonable default that may not exist (readers already handle the
    missing file case). Import time side‑effects are avoided.
    """
    override = os.getenv("PIPELINE_STATE_FILE")
    if override:
        return Path(override)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pipeline_runs.json"
        if candidate.exists():
            return candidate

    # Fallback: assume the file is a few levels up from this module.
    fallback_idx = min(4, len(here.parents) - 1)
    return here.parents[fallback_idx] / "pipeline_runs.json"


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

def _load_runs(limit: int = 50) -> list[dict]:
    """Load the most recent runs from the state file, bounded by *limit*."""
    if not _STATE_FILE.exists():
        return []
    try:
        data = json.loads(_STATE_FILE.read_text())
        if not isinstance(data, list):
            return []
        return sorted(data, key=lambda r: r.get("started_at", ""), reverse=True)[:limit]
    except Exception:
        return []

def _get_pipeline_def(pipeline_name: str) -> dict:
    """Return the pipeline definition for *pipeline_name* or an empty dict."""
    return PIPELINE_DEFS.get(pipeline_name, {})

def _index_actual_stages(stages: list[dict]) -> dict[str, dict]:
    """Create a mapping from stage name to its actual result dict."""
    return {stage["name"]: stage for stage in stages if "name" in stage}

def _build_merged_stages(
    defn_stages: list[dict],
    actual_index: dict[str, dict],
) -> list[dict]:
    """Merge definition order with actual stage data, inserting pending status where missing."""
    merged: list[dict] = []
    for sdef in defn_stages:
        name = sdef["name"]
        if name in actual_index:
            merged.append({**sdef, **actual_index[name]})
        else:
            merged.append({**sdef, "status": "pending"})
    return merged

def _append_extra_stages(
    merged: list[dict],
    original_stages: list[dict],
    known_names: set[str],
) -> None:
    """Append any stages present in *original_stages* that are not part of the definition."""
    for stage in original_stages:
        if stage.get("name") not in known_names:
            merged.append(stage)

def _enrich_run(run: dict) -> dict:
    """Add stage definitions so the frontend knows the expected stage order."""
    pipeline_name = run.get("pipeline", "")
    defn = _get_pipeline_def(pipeline_name)

    # Preserve original run dict while allowing modifications.
    enriched = dict(run)

    # Index actual stages for quick lookup.
    actual_index = _index_actual_stages(enriched.get("stages", []))

    # Merge definition order with actual data.
    merged = _build_merged_stages(defn.get("stages", []), actual_index)

    # Append any extra stages that were not defined.
    known_stage_names = {s["name"] for s in defn.get("stages", [])}
    _append_extra_stages(merged, enriched.get("stages", []), known_stage_names)

    enriched["stages"] = merged
    enriched["pipeline_label"] = defn.get("label", pipeline_name)
    return enriched

@router.get("/status")
def pipeline_status(
    pipeline: Optional[str] = Query(None),
    desk: Optional[str] = Query(None),
    limit: int = Query(20, le=50),
):
    """Return recent pipeline runs, optionally filtered by pipeline name or desk."""
    runs = _load_runs(limit * 2)
    if pipeline:
        runs = [r for r in runs if r.get("pipeline") == pipeline]
    if desk:
        runs = [r for r in runs if r.get("desk") == desk]
    return [_enrich_run(r) for r in runs[:limit]]

@router.get("/status/latest")
def pipeline_status_latest():
    """Return the most recent run for each pipeline type."""
    runs = _load_runs(100)
    seen: set[str] = set()
    latest: list[dict] = []
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