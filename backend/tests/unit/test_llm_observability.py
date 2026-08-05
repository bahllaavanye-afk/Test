"""Tests for the LLM cascade observability: metrics logging + cascade_status probe."""
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, validator

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import llm_common as L  # noqa: E402


class MetricRecord(BaseModel):
    """Schema representing a single metric log entry for an LLM provider request."""

    provider: str = Field(
        ...,
        description="Name of the LLM provider.",
        example="groq",
    )
    ok: bool = Field(
        ...,
        description="Flag indicating whether the request succeeded.",
        example=True,
    )
    ms: int = Field(
        ...,
        ge=0,
        description="Latency of the request in milliseconds.",
        example=123,
    )
    error: str | None = Field(
        default=None,
        description="Optional error message when the request fails.",
        example="all providers failed",
    )

    @validator("provider")
    def provider_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("provider must be a non‑empty string")
        return v


class ProviderStatus(BaseModel):
    """Status information for a single LLM provider."""

    has_key: bool = Field(
        ...,
        description="Indicates whether the required API key environment variable is set.",
        example=True,
    )
    # Additional provider‑specific fields can be added here if needed.


class CascadeStatus(BaseModel):
    """Aggregated status report for the LLM cascade."""

    providers: dict[str, ProviderStatus] = Field(
        ...,
        description="Mapping from provider name to its status information.",
        example={"groq": {"has_key": True}},
    )
    healthy: bool = Field(
        ...,
        description="Overall health flag for the cascade (True if any provider is operational).",
        example=False,
    )
    working: list[str] = Field(
        ...,
        description="List of provider names that responded successfully during probing.",
        example=["groq"],
    )


def test_record_metric_appends_jsonl(tmp_path, monkeypatch):
    f = tmp_path / "m.jsonl"
    monkeypatch.setattr(L, "_METRICS_FILE", f)
    L._record_metric("groq", True, 123)
    L._record_metric("none", False, 50, "all providers failed")
    lines = f.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["provider"] == "groq" and rec["ok"] is True and rec["ms"] == 123
    assert json.loads(lines[1])["ok"] is False


def test_record_metric_never_raises(monkeypatch):
    # Unwritable path must be swallowed, not crash the agent.
    monkeypatch.setattr(L, "_METRICS_FILE", Path("/nonexistent-dir/xyz/m.jsonl"))
    L._record_metric("groq", True, 1)  # no exception = pass


def test_cascade_status_reports_key_presence(monkeypatch):
    for p in L._PROVIDERS:
        monkeypatch.delenv(p["key_env"], raising=False)
        if p.get("key_env_alt"):
            monkeypatch.delenv(p["key_env_alt"], raising=False)
    first = L._PROVIDERS[0]
    monkeypatch.setenv(first["key_env"], "k")
    st = L.cascade_status(probe=False)  # probe=False → no network
    assert st["providers"][first["name"]]["has_key"] is True
    assert st["healthy"] is False  # nothing probed/working
    assert isinstance(st["working"], list)