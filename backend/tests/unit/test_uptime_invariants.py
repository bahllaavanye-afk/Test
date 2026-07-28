"""Enforce the no-downtime invariants so they can't silently regress.

Answers "no downtime?" — kept alive by TWO independent layers: an external
GitHub-Actions cron pinging /health (survives Render sleep), and an in-app
self-ping. These static checks fail loudly if either is removed.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, validator


ROOT = Path(__file__).resolve().parents[3]


class KeepAliveConfig(BaseModel):
    """Schema representing the external keep‑alive workflow configuration."""

    cron: str = Field(
        ...,
        description="Cron expression that defines the schedule for the keep‑alive job.",
        example="*/5 * * * *",
    )
    endpoint: str = Field(
        ...,
        description="HTTP endpoint that the keep‑alive job should ping.",
        example="/health",
    )

    @validator("cron")
    def cron_must_run_every_five_minutes(cls, v: str) -> str:
        """Ensure the cron expression includes a five‑minute interval."""
        if "*/5" not in v:
            raise ValueError("cron expression must contain '*/5' to run every 5 minutes")
        return v

    @validator("endpoint")
    def endpoint_must_be_health(cls, v: str) -> str:
        """The keep‑alive job must target the health endpoint."""
        if "/health" not in v:
            raise ValueError("endpoint must contain '/health'")
        return v


class SchedulerConfig(BaseModel):
    """Schema representing the scheduler script expectations."""

    jobs: list[str] = Field(
        ...,
        description="List of job identifiers that should be registered in the scheduler.",
        example=["self_ping", "bot_runner_ignition", "bot_exit_checker"],
    )
    render_url_env: str = Field(
        ...,
        description="Environment variable name used to target the public Render URL.",
        example="RENDER_EXTERNAL_URL",
    )

    @validator("jobs", each_item=True)
    def job_names_must_be_non_empty(cls, v: str) -> str:
        """Each job name must be a non‑empty string."""
        if not v.strip():
            raise ValueError("job name cannot be empty")
        return v

    @validator("render_url_env")
    def render_url_env_must_be_present(cls, v: str) -> str:
        """Validate that the expected environment variable name is provided."""
        if not v:
            raise ValueError("render_url_env must be a non‑empty string")
        return v


def test_external_keepalive_pings_health_every_5_min():
    ka = (ROOT / ".github" / "workflows" / "keep-alive.yml").read_text()
    assert 'cron: "*/5 * * * *"' in ka or "*/5 * * * *" in ka, "keep-alive must run every 5 minutes"
    assert "/health" in ka, "keep-alive must hit the /health endpoint"


def test_in_app_self_ping_job_registered():
    sched = (ROOT / "backend" / "app" / "tasks" / "scheduler.py").read_text()
    assert '"self_ping"' in sched or "self_ping" in sched, "scheduler must register the self_ping job"
    assert "RENDER_EXTERNAL_URL" in sched, "self-ping must target the public Render URL"


def test_health_endpoint_exists():
    main = (ROOT / "backend" / "app" / "main.py").read_text()
    assert '"/health"' in main, "backend must expose a /health endpoint"


def test_bot_runner_and_exit_checker_are_scheduled():
    """No-downtime for trading: the runner + exit checker must be wired in."""
    sched = (ROOT / "backend" / "app" / "tasks" / "scheduler.py").read_text()
    assert "bot_runner_ignition" in sched, "bot runner ignition job must exist"
    assert "bot_exit_checker" in sched, "bot exit checker job must exist (positions must close)"