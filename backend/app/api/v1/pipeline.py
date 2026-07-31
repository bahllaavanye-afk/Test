"""
Pipeline status API — reads pipeline_runs.json written by GitHub Actions scripts.
No database needed: the JSON file is the source of truth.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# --------------------------------------------------------------------------- #
# Helper: Resolve the location of the pipeline state file.
# --------------------------------------------------------------------------- #
def _resolve_state_file() -> Path:
    """Locate ``pipeline_runs.json`` without assuming a fixed directory depth.

    The previous implementation used ``parents[5]`` which could raise ``IndexError``
    on environments with a different repository layout. This function searches
    upward from the current file for the JSON file, respects the
    ``PIPELINE_STATE_FILE`` environment variable, and falls back to a reasonable
    default without raising at import time.
    """
    override = os.environ.get("PIPELINE_STATE_FILE")
    if override:
        return Path(override)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pipeline_runs.json"
        if candidate.exists():
            return candidate

    # Fallback: assume the file lives a few levels up from this module.
    fallback_index = min(4, len(here.parents) - 1)
    return here.parents[fallback_index] / "pipeline_runs.json"


_STATE_FILE = _resolve_state_file()

# --------------------------------------------------------------------------- #
# Static pipeline definitions – consumed by the frontend.
# --------------------------------------------------------------------------- #
PIPELINE_DEFS = {
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


# --------------------------------------------------------------------------- #
# Internal utilities.
# --------------------------------------------------------------------------- #
def _load_runs(limit: int = 50) -> List[dict]:
    """Load the most recent pipeline runs from the JSON state file.

    Returns an empty list if the file is missing or malformed.
    """
    if not _STATE_FILE.exists():
        return []
    try:
        data = json.loads(_STATE_FILE.read_text())
        if not isinstance(data, list):
            return []
        # Sort by ``started_at`` descending and truncate to ``limit``.
        return sorted(data, key=lambda r: r.get("started_at", ""), reverse=True)[:limit]
    except Exception:
        return []


def _enrich_run(run: dict) -> dict:
    """Add static stage definitions to a run for frontend consumption.

    The enrichment merges the static definition order with the actual stage
    results, preserving any additional stages that may appear in the run.
    """
    pipeline = run.get("pipeline", "")
    defn = PIPELINE_DEFS.get(pipeline, {})
    stage_order = [s["name"] for s in defn.get("stages", [])]

    # Map actual stage results by name for quick lookup.
    actual: dict[str, dict] = {s["name"]: s for s in run.get("stages", [])}

    merged: List[dict] = []
    for sdef in defn.get("stages", []):
        sname = sdef["name"]
        if sname in actual:
            merged.append({**sdef, **actual[sname]})
        else:
            merged.append({**sdef, "status": "pending"})

    # Append any stages present in the run but not defined statically.
    for s in run.get("stages", []):
        if s["name"] not in stage_order:
            merged.append(s)

    enriched = dict(run)
    enriched["stages"] = merged
    enriched["pipeline_label"] = defn.get("label", pipeline)
    return enriched


def _is_signal_valid(run: dict) -> bool:
    """Apply tightened entry and exit criteria to a Desk Trading run.

    Entry filters:
        * ``signal_strength`` must be >= 0.7.
        * At least one confirmation indicator (e.g., ``price_action`` or ``volume``)
          must be present in the ``confirmations`` list.

    Exit filters:
        * ``order_execution`` stage must contain a non‑empty ``exit_reason``.

    Returns ``True`` if the run satisfies all criteria, ``False`` otherwise.
    """
    if run.get("pipeline") != "desk_trading":
        return True  # Non‑desk pipelines are not subject to these checks.

    # Locate stages of interest.
    stages = {s["name"]: s for s in run.get("stages", [])}
    signal_stage = stages.get("signal_generation", {})
    exec_stage = stages.get("order_execution", {})

    # Entry criteria.
    strength = signal_stage.get("signal_strength")
    if not isinstance(strength, (int, float)) or strength < 0.7:
        return False

    confirmations = signal_stage.get("confirmations", [])
    if not isinstance(confirmations, list) or not any(
        c in {"price_action", "volume"} for c in confirmations
    ):
        return False

    # Exit criteria.
    exit_reason = exec_stage.get("exit_reason")
    if not isinstance(exit_reason, str) or not exit_reason.strip():
        return False

    return True


# --------------------------------------------------------------------------- #
# API endpoints.
# --------------------------------------------------------------------------- #
@router.get("/status")
def pipeline_status(
    pipeline: Optional[str] = Query(None),
    desk: Optional[str] = Query(None),
    limit: int = Query(20, le=50),
    valid_only: bool = Query(False, description="Return only runs that pass signal quality checks."),
) -> List[dict]:
    """Return recent pipeline runs, optionally filtered by pipeline or desk.

    Parameters
    ----------
    pipeline: Optional[str]
        Filter runs by the ``pipeline`` name.
    desk: Optional[str]
        Filter runs by the ``desk`` identifier.
    limit: int
        Maximum number of runs to return (capped at 50).
    valid_only: bool
        If ``True``, only runs that satisfy tightened signal quality criteria are
        included. This is primarily useful for the Desk Trading pipeline.
    """
    # Load a superset to allow post‑filtering before truncating.
    runs = _load_runs(limit * 2)

    if pipeline:
        runs = [r for r in runs if r.get("pipeline") == pipeline]
    if desk:
        runs = [r for r in runs if r.get("desk") == desk]
    if valid_only:
        runs = [r for r in runs if _is_signal_valid(r)]

    # Preserve the original ordering (newest first) and respect the limit.
    return [_enrich_run(r) for r in runs[:limit]]


@router.get("/status/latest")
def pipeline_status_latest() -> List[dict]:
    """Return the most recent run for each unique ``pipeline:desk`` combination."""
    runs = _load_runs(100)
    seen: set[str] = set()
    latest: List[dict] = []
    for run in runs:
        key = f"{run.get('pipeline')}:{run.get('desk', '')}"
        if key not in seen:
            seen.add(key)
            latest.append(_enrich_run(run))
    return latest


@router.get("/status/{run_id}")
def pipeline_run_detail(run_id: str) -> dict:
    """Return full detail for a specific pipeline run."""
    for run in _load_runs(100):
        if run.get("run_id") == run_id:
            return _enrich_run(run)
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")


@router.get("/definitions")
def pipeline_definitions() -> dict:
    """Return static pipeline stage definitions for the frontend."""
    return PIPELINE_DEFS