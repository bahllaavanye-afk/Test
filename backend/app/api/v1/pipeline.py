"""
Pipeline status API — reads pipeline_runs.json written by GitHub Actions scripts.
No database needed: the JSON file is the source of truth.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum

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


class StageStatus(str, Enum):
    """Allowed status values for a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class StageDefinition(BaseModel):
    """Static definition of a pipeline stage."""

    name: str = Field(..., description="Internal identifier for the stage.", example="data_fetch")
    label: str = Field(..., description="Human‑readable label shown in UI.", example="Data Fetch")
    channel: Optional[str] = Field(
        None,
        description="Slack channel to post notifications to, if any.",
        example="#squad-data",
    )

    @validator("channel")
    def channel_must_start_with_hash(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith("#"):
            raise ValueError("channel must start with '#'")
        return v


class StageResult(StageDefinition):
    """Result of a pipeline stage execution."""

    status: StageStatus = Field(
        StageStatus.PENDING,
        description="Current execution status of the stage.",
        example="success",
    )
    started_at: Optional[str] = Field(
        None,
        description="ISO‑8601 timestamp when the stage started.",
        example="2024-01-01T12:00:00Z",
    )
    completed_at: Optional[str] = Field(
        None,
        description="ISO‑8601 timestamp when the stage finished.",
        example="2024-01-01T12:05:00Z",
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional free‑form details emitted by the stage.",
        example={"records_processed": 12345},
    )


class PipelineDefinition(BaseModel):
    """Static definition for a pipeline type."""

    label: str = Field(..., description="Human readable pipeline label.", example="ML Experiments")
    stages: List[StageDefinition] = Field(
        ...,
        description="Ordered list of stage definitions for this pipeline.",
    )


class PipelineRun(BaseModel):
    """Aggregated information for a single pipeline execution."""

    run_id: str = Field(..., description="Unique identifier for the pipeline run.", example="run-2024-01-01-001")
    pipeline: str = Field(..., description="Internal pipeline name.", example="ml_experiments")
    pipeline_label: Optional[str] = Field(
        None,
        description="Human readable label for the pipeline (derived from definitions).",
        example="ML Experiments",
    )
    desk: Optional[str] = Field(
        None,
        description="Desk identifier if the pipeline is desk‑specific.",
        example="desk_a",
    )
    started_at: Optional[str] = Field(
        None,
        description="ISO‑8601 timestamp when the pipeline started.",
        example="2024-01-01T12:00:00Z",
    )
    completed_at: Optional[str] = Field(
        None,
        description="ISO‑8601 timestamp when the pipeline completed.",
        example="2024-01-01T12:30:00Z",
    )
    stages: List[StageResult] = Field(
        ...,
        description="List of stage results in the order defined for the pipeline.",
    )

    @root_validator(pre=True)
    def ensure_stage_order(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Guarantee that stages are ordered according to the static definition."""
        pipeline = values.get("pipeline", "")
        defn = PIPELINE_DEFS.get(pipeline, {})
        stage_order = [s["name"] for s in defn.get("stages", [])]
        actual = {s["name"]: s for s in values.get("stages", [])}
        merged: List[Dict[str, Any]] = []
        for name in stage_order:
            if name in actual:
                merged.append(actual[name])
            else:
                merged.append({"name": name, "status": StageStatus.PENDING})
        # Append any extra stages not defined
        extra = [s for s in values.get("stages", []) if s["name"] not in stage_order]
        merged.extend(extra)
        values["stages"] = merged
        return values


def _load_runs(limit: int = 50) -> List[Dict[str, Any]]:
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
    """Add stage definitions so the frontend knows the expected stage order."""
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


@router.get("/status", response_model=List[PipelineRun])
def pipeline_status(
    pipeline: Optional[str] = Query(None, description="Filter by pipeline name.", example="ml_experiments"),
    desk: Optional[str] = Query(None, description="Filter by desk identifier.", example="desk_a"),
    limit: int = Query(20, le=50, description="Maximum number of runs to return (max 50).", example=20),
):
    """Return recent pipeline runs, optionally filtered by pipeline name or desk."""
    runs = _load_runs(limit * 2)
    if pipeline:
        runs = [r for r in runs if r.get("pipeline") == pipeline]
    if desk:
        runs = [r for r in runs if r.get("desk") == desk]
    return [_enrich_run(r) for r in runs[:limit]]


@router.get("/status/latest", response_model=List[PipelineRun])
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


@router.get("/status/{run_id}", response_model=PipelineRun)
def pipeline_run_detail(run_id: str):
    """Return full detail for a specific pipeline run."""
    for run in _load_runs(100):
        if run.get("run_id") == run_id:
            return _enrich_run(run)
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")


@router.get("/definitions", response_model=Dict[str, PipelineDefinition])
def pipeline_definitions():
    """Return static pipeline stage definitions for the frontend."""
    return PIPELINE_DEFS