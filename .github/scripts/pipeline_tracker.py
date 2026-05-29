"""
QuantEdge Pipeline Tracker — stage-level status tracking for GitHub Actions.

Usage in any runner script:
    from pipeline_tracker import PipelineTracker, Stage

    tracker = PipelineTracker(pipeline="ml_experiments", desk="equities")
    tracker.start()

    with tracker.stage(Stage.DATA_FETCH, "Fetch OHLCV", channel="#squad-data"):
        df = yf.download(...)
        tracker.set_output(n_bars=len(df), symbol=symbol)

    with tracker.stage(Stage.BACKTESTING, "Run backtests", channel="#ml-experiments"):
        ...

    tracker.complete()

State is written to pipeline_runs.json at the repo root and optionally
posted to Slack per-stage.
"""
from __future__ import annotations

import json
import os
import time
import traceback
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Generator, Optional

REPO_ROOT    = Path(__file__).resolve().parents[2]
STATE_FILE   = REPO_ROOT / "pipeline_runs.json"
MAX_RUNS     = 50          # keep last N runs in the JSON
SLACK_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")
RUN_ID       = os.environ.get("GITHUB_RUN_ID", f"local-{int(time.time())}")
WORKFLOW     = os.environ.get("GITHUB_WORKFLOW", "local")
REPO         = os.environ.get("GITHUB_REPOSITORY", "unknown/unknown")
BRANCH       = os.environ.get("GITHUB_REF_NAME", "local")
TRIGGER      = os.environ.get("GITHUB_EVENT_NAME", "manual")
RUN_URL      = f"https://github.com/{REPO}/actions/runs/{RUN_ID}"


# ─── Stage constants ───────────────────────────────────────────────────────────

class Stage:
    # Shared
    DATA_FETCH          = "data_fetch"
    CACHE_CHECK         = "cache_check"
    # ML pipeline
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING      = "model_training"
    BACKTESTING         = "backtesting"
    EVALUATION          = "evaluation"
    SLACK_REPORT        = "slack_report"
    COMMIT_RESULTS      = "commit_results"
    # Trading pipeline
    MARKET_STATUS       = "market_status"
    SIGNAL_GENERATION   = "signal_generation"
    RISK_CHECK          = "risk_check"
    ORDER_EXECUTION     = "order_execution"
    FILL_TRACKING       = "fill_tracking"
    PNL_SNAPSHOT        = "pnl_snapshot"
    # Agent pipeline
    AGENT_DISPATCH      = "agent_dispatch"
    AGENT_POSTS         = "agent_posts"


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED  = "failed"
    SKIPPED = "skipped"


# ─── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StageRecord:
    name:         str
    label:        str
    status:       str   = Status.PENDING
    started_at:   Optional[str] = None
    completed_at: Optional[str] = None
    duration_s:   Optional[float] = None
    output:       dict  = field(default_factory=dict)
    error:        Optional[str] = None
    channel:      Optional[str] = None


@dataclass
class PipelineRunRecord:
    run_id:       str
    pipeline:     str
    desk:         Optional[str]
    branch:       str
    triggered_by: str
    started_at:   str
    completed_at: Optional[str] = None
    status:       str = Status.RUNNING
    stages:       list[StageRecord] = field(default_factory=list)
    run_url:      str = ""


# ─── Slack helpers ─────────────────────────────────────────────────────────────

def _slack_post(channel: str, text: str) -> None:
    if not SLACK_TOKEN:
        print(f"  [pipeline] no SLACK_BOT_TOKEN — skip: {text[:80]}", flush=True)
        return
    payload = json.dumps({"channel": channel, "text": text}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=payload,
        headers={"Authorization": f"Bearer {SLACK_TOKEN}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            body = json.loads(r.read())
            if not body.get("ok"):
                print(f"  [pipeline] Slack error: {body.get('error')}", flush=True)
    except Exception as exc:
        print(f"  [pipeline] Slack post failed: {exc}", flush=True)


# ─── Main tracker ──────────────────────────────────────────────────────────────

class PipelineTracker:
    """
    Context-manager-based pipeline tracker.

    with PipelineTracker("ml_experiments") as t:
        with t.stage(Stage.DATA_FETCH, "Fetch OHLCV", channel="#squad-data"):
            ...
            t.set_output(n_bars=252)
    """

    def __init__(self, pipeline: str, desk: Optional[str] = None):
        self.pipeline  = pipeline
        self.desk      = desk
        self._record   = PipelineRunRecord(
            run_id       = RUN_ID,
            pipeline     = pipeline,
            desk         = desk,
            branch       = BRANCH,
            triggered_by = TRIGGER,
            started_at   = _now(),
            run_url      = RUN_URL,
        )
        self._current_stage: Optional[StageRecord] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __enter__(self) -> "PipelineTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self._record.status = Status.FAILED
        else:
            self._record.status = Status.SUCCESS
        self.complete()
        return False   # don't suppress exceptions

    def start(self) -> None:
        self._record.started_at = _now()
        self._record.status     = Status.RUNNING
        self._save()
        label = f"{self.pipeline}" + (f"/{self.desk}" if self.desk else "")
        print(f"\n[Pipeline] {label} — started ({BRANCH} · {TRIGGER})", flush=True)

    def complete(self, status: str = Status.SUCCESS) -> None:
        if self._record.status == Status.RUNNING:
            self._record.status = status
        self._record.completed_at = _now()
        self._save()
        label = f"{self.pipeline}" + (f"/{self.desk}" if self.desk else "")
        icon  = "✅" if self._record.status == Status.SUCCESS else "❌"
        print(f"\n[Pipeline] {icon} {label} — {self._record.status}", flush=True)

    # ── Stage context manager ─────────────────────────────────────────────────

    @contextmanager
    def stage(
        self,
        name: str,
        label: str,
        channel: Optional[str] = None,
        skip_if: bool = False,
    ) -> Generator[StageRecord, None, None]:
        sr = StageRecord(name=name, label=label, channel=channel)
        self._record.stages.append(sr)
        self._current_stage = sr

        if skip_if:
            sr.status = Status.SKIPPED
            self._save()
            yield sr
            return

        sr.status     = Status.RUNNING
        sr.started_at = _now()
        t0 = time.monotonic()
        self._save()

        # Slack: stage started
        if channel:
            _slack_post(channel, f":hourglass_flowing_sand: *{label}* started  |  pipeline `{self.pipeline}`  |  <{RUN_URL}|run →>")

        print(f"\n  [stage] {label} — running", flush=True)

        try:
            yield sr
            sr.status       = Status.SUCCESS
            sr.duration_s   = round(time.monotonic() - t0, 2)
            sr.completed_at = _now()
            self._save()
            # Slack: stage done
            summary = _format_output(sr.output)
            if channel:
                _slack_post(channel,
                    f":white_check_mark: *{label}* completed in `{sr.duration_s}s`"
                    + (f"  —  {summary}" if summary else "")
                    + f"  |  <{RUN_URL}|run →>")
            print(f"  [stage] ✓ {label} — {sr.duration_s}s  {summary}", flush=True)

        except Exception as exc:
            sr.status       = Status.FAILED
            sr.duration_s   = round(time.monotonic() - t0, 2)
            sr.completed_at = _now()
            sr.error        = f"{type(exc).__name__}: {exc}"
            self._record.status = Status.FAILED
            self._save()
            if channel:
                _slack_post(channel,
                    f":x: *{label}* FAILED after `{sr.duration_s}s`\n"
                    f"```{sr.error}```  |  <{RUN_URL}|run →>")
            print(f"  [stage] ✗ {label} — {sr.error}", flush=True)
            traceback.print_exc()
            raise

        finally:
            self._current_stage = None

    def set_output(self, **kwargs) -> None:
        """Set output metadata on the current stage."""
        if self._current_stage is not None:
            self._current_stage.output.update(kwargs)
            self._save()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        existing: list[dict] = []
        if STATE_FILE.exists():
            try:
                existing = json.loads(STATE_FILE.read_text())
            except Exception:
                existing = []

        # Replace or append current run
        updated = [r for r in existing if r.get("run_id") != self._record.run_id]
        updated.append(_to_dict(self._record))
        # Keep newest MAX_RUNS runs
        updated = sorted(updated, key=lambda r: r.get("started_at", ""), reverse=True)[:MAX_RUNS]
        STATE_FILE.write_text(json.dumps(updated, indent=2, default=str))


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_output(output: dict) -> str:
    if not output:
        return ""
    parts = []
    for k, v in output.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:+.3f}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts[:4])


def _to_dict(obj) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    return obj


# ─── Read helpers (used by API / Slack agents) ─────────────────────────────────

def load_runs(limit: int = 20) -> list[dict]:
    if not STATE_FILE.exists():
        return []
    try:
        runs = json.loads(STATE_FILE.read_text())
        return sorted(runs, key=lambda r: r.get("started_at", ""), reverse=True)[:limit]
    except Exception:
        return []


def latest_run(pipeline: Optional[str] = None, desk: Optional[str] = None) -> Optional[dict]:
    for run in load_runs():
        if pipeline and run.get("pipeline") != pipeline:
            continue
        if desk and run.get("desk") != desk:
            continue
        return run
    return None
