"""
Pipeline status API — reads pipeline_runs.json written by GitHub Actions scripts.
No database needed: the JSON file is the source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

_STATE_FILE = Path(__file__).resolve().parents[5] / "pipeline_runs.json"

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


# --------------------------------------------------------------------------- #
# Unit tests for edge‑case behavior
# --------------------------------------------------------------------------- #

import pytest
from datetime import datetime, timedelta


def _write_state(tmp_path: Path, data) -> Path:
    """Helper to write JSON data to a temporary state file and return its path."""
    p = tmp_path / "pipeline_runs.json"
    p.write_text(json.dumps(data))
    return p


def test_load_runs_file_missing(monkeypatch, tmp_path):
    """When the state file does not exist, _load_runs should return an empty list."""
    monkeypatch.setattr(
        "backend.app.api.v1.pipeline._STATE_FILE",
        tmp_path / "nonexistent.json",
    )
    assert _load_runs() == []


def test_load_runs_invalid_json(monkeypatch, tmp_path):
    """Invalid JSON content should be treated as an empty result."""
    bad_file = tmp_path / "pipeline_runs.json"
    bad_file.write_text("not a json")
    monkeypatch.setattr(
        "backend.app.api.v1.pipeline._STATE_FILE",
        bad_file,
    )
    assert _load_runs() == []


def test_load_runs_respects_limit_and_sort(monkeypatch, tmp_path):
    """_load_runs must sort by started_at descending and respect the limit."""
    now = datetime.utcnow()
    runs = [
        {"run_id": f"id_{i}", "started_at": (now - timedelta(minutes=i)).isoformat()}
        for i in range(10)
    ]
    state_path = _write_state(tmp_path, runs)
    monkeypatch.setattr(
        "backend.app.api.v1.pipeline._STATE_FILE",
        state_path,
    )
    result = _load_runs(limit=5)
    assert len(result) == 5
    # Verify ordering: most recent first
    expected_ids = [f"id_{i}" for i in range(5)]
    assert [r["run_id"] for r in result] == expected_ids


def test_enrich_run_unknown_pipeline(monkeypatch):
    """When a pipeline is not defined, enrich should preserve original data."""
    unknown = {"pipeline": "unknown_pipe", "stages": [{"name": "custom", "status": "done"}]}
    enriched = _enrich_run(unknown)
    assert enriched["pipeline_label"] == "unknown_pipe"
    assert enriched["stages"] == unknown["stages"]


def test_pipeline_status_latest_deduplication(monkeypatch, tmp_path):
    """Only the newest run per pipeline+desk pair should be returned."""
    now = datetime.utcnow()
    runs = [
        {
            "run_id": "first",
            "pipeline": "ml_experiments",
            "desk": "alpha",
            "started_at": (now - timedelta(hours=2)).isoformat(),
        },
        {
            "run_id": "second",
            "pipeline": "ml_experiments",
            "desk": "alpha",
            "started_at": (now - timedelta(hours=1)).isoformat(),
        },
        {
            "run_id": "third",
            "pipeline": "desk_trading",
            "desk": "beta",
            "started_at": now.isoformat(),
        },
    ]
    state_path = _write_state(tmp_path, runs)
    monkeypatch.setattr(
        "backend.app.api.v1.pipeline._STATE_FILE",
        state_path,
    )
    latest = pipeline_status_latest()
    # Expect two entries: the most recent for each unique key
    ids = {run["run_id"] for run in latest}
    assert ids == {"second", "third"}
    # Ensure ordering respects the original sorted order (most recent first)
    assert latest[0]["run_id"] == "third"
    assert latest[1]["run_id"] == "second"