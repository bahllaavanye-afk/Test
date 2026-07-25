"""
Pipeline status API — reads pipeline_runs.json written by GitHub Actions scripts.
No database needed: the JSON file is the source of truth.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

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

PIPELINE_DEFS = {
    "ml_experiments": {
        "label": "ML Experiments",
        "stages": [
            {"name": "data_fetch",          "label": "Data Fetch",          "channel": "#squad-data"},
            {"name": "cache_check",         "label": "Cache Check",         "channel": "#squad-data"},
            {"name": "feature_engineering", "label": "Feature Engineering", "channel": "#alpha-research"},
            {"name": "backtesting",         "label": "Backtesting",         "channel": "#ml-experiments"},
            {"name": "evaluation",          "label": "Evaluation",          "channel": "#ml-experiments"},
            {"name": "slack_report",        "label": "Slack Report",        "channel": "#ml-experiments"},
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
            {"name": "agent_posts",   "label": "Slack Posts",   "channel": "#engineering"},
        ],
    },
}


def _load_runs(limit: int = 50) -> list[dict]:
    if not _STATE_FILE.exists():
        return []
    try:
        data = json.loads(_STATE_FILE.read_text())
        if not isinstance(data, list):
            return []
        return sorted(data, key=lambda r: r.get("started_at", ""), reverse=True)[:limit]
    except Exception:
        return []


def _enrich_run(run: dict) -> dict:
    """Add stage definitions so the frontend knows the expected stage order."""
    pipeline = run.get("pipeline", "")
    defn = PIPELINE_DEFS.get(pipeline, {})
    stage_order = [s["name"] for s in defn.get("stages", [])]
    run = dict(run)

    # Index actual stage results by name
    actual: dict[str, dict] = {s["name"]: s for s in run.get("stages", [])}

    # Build merged list: definition order, with actual data filled in
    merged = []
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
    runs    = _load_runs(100)
    seen:   set[str] = set()
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


# ----------------------------------------------------------------------
# Unit tests for edge‑case behavior
# ----------------------------------------------------------------------
def test_resolve_state_file_env_override():
    """When PIPELINE_STATE_FILE env var is set, the function should return it verbatim."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
        os.environ["PIPELINE_STATE_FILE"] = str(tmp_path)
        try:
            resolved = _resolve_state_file()
            assert resolved == tmp_path
        finally:
            del os.environ["PIPELINE_STATE_FILE"]
            tmp_path.unlink(missing_ok=True)


def test_load_runs_malformed_json(tmp_path):
    """If the JSON file does not contain a list, _load_runs should return an empty list."""
    file_path = tmp_path / "pipeline_runs.json"
    file_path.write_text('{"not": "a list"}')
    global _STATE_FILE
    _STATE_FILE = file_path
    runs = _load_runs()
    assert runs == []


def test_load_runs_limit_and_sorting(tmp_path):
    """_load_runs must respect the limit argument and sort by started_at descending."""
    file_path = tmp_path / "pipeline_runs.json"
    data = [
        {"run_id": "a", "started_at": "2023-01-01"},
        {"run_id": "b", "started_at": "2023-01-02"},
        {"run_id": "c", "started_at": "2023-01-03"},
    ]
    file_path.write_text(json.dumps(data))
    global _STATE_FILE
    _STATE_FILE = file_path
    runs = _load_runs(limit=2)
    assert len(runs) == 2
    assert runs[0]["run_id"] == "c"
    assert runs[1]["run_id"] == "b"


def test_enrich_run_stage_merging():
    """_enrich_run should merge defined stages with actual data, add pending status for missing stages,
    and preserve extra stages not in the definition."""
    run = {
        "pipeline": "ml_experiments",
        "run_id": "test123",
        "stages": [
            {"name": "data_fetch", "status": "complete"},
            {"name": "extra_stage", "status": "unknown"},
        ],
    }
    enriched = _enrich_run(run)
    # Verify order: all defined stages first, then the extra stage
    defined_names = [s["name"] for s in PIPELINE_DEFS["ml_experiments"]["stages"]]
    result_names = [s["name"] for s in enriched["stages"]]
    assert result_names == defined_names + ["extra_stage"]
    # Verify that a defined stage not present in the original gets a pending status
    cache_stage = next(s for s in enriched["stages"] if s["name"] == "cache_check")
    assert cache_stage["status"] == "pending"
    # Verify that the extra stage is unchanged
    extra = next(s for s in enriched["stages"] if s["name"] == "extra_stage")
    assert extra["status"] == "unknown"
    # Verify pipeline label is set correctly
    assert enriched["pipeline_label"] == PIPELINE_DEFS["ml_experiments"]["label"]