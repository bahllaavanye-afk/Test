"""Tests for the LLM cascade observability: metrics logging + cascade_status probe."""
import json
import sys
from pathlib import Path
from typing import List

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import llm_common as L  # noqa: E402


def _read_jsonl(file_path: Path) -> List[dict]:
    """Read a JSON Lines file and return a list of parsed dictionaries."""
    content = file_path.read_text().strip()
    if not content:
        return []
    return [json.loads(line) for line in content.splitlines()]


def test_record_metric_appends_jsonl(tmp_path: Path, monkeypatch) -> None:
    """Ensure that _record_metric appends well‑formed JSON lines."""
    metrics_file = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(L, "_METRICS_FILE", metrics_file)

    L._record_metric("groq", True, 123)
    L._record_metric("none", False, 50, "all providers failed")

    records = _read_jsonl(metrics_file)
    assert len(records) == 2

    first = records[0]
    assert first["provider"] == "groq"
    assert first["ok"] is True
    assert first["ms"] == 123

    second = records[1]
    assert second["provider"] == "none"
    assert second["ok"] is False
    assert second["error"] == "all providers failed"


def test_record_metric_never_raises(monkeypatch) -> None:
    """Unwritable path must be swallowed; the call should never raise."""
    # Point to a location that cannot be created.
    monkeypatch.setattr(L, "_METRICS_FILE", Path("/nonexistent-dir/xyz/m.jsonl"))
    # No exception means the function handled the error gracefully.
    L._record_metric("groq", True, 1)


def test_cascade_status_reports_key_presence(monkeypatch) -> None:
    """Validate that cascade_status correctly reflects environment key presence."""
    # Ensure no provider environment variables are set.
    for provider in L._PROVIDERS:
        monkeypatch.delenv(provider["key_env"], raising=False)
        if provider.get("key_env_alt"):
            monkeypatch.delenv(provider["key_env_alt"], raising=False)

    # Set a key for the first provider only.
    first = L._PROVIDERS[0]
    monkeypatch.setenv(first["key_env"], "dummy_key")

    status = L.cascade_status(probe=False)  # probe=False → no network calls
    providers_status = status["providers"]
    assert providers_status[first["name"]]["has_key"] is True

    # All other providers should report has_key=False.
    for provider in L._PROVIDERS[1:]:
        assert providers_status[provider["name"]]["has_key"] is False

    assert status["healthy"] is False  # No providers have been probed/working yet.
    assert isinstance(status["working"], list)


def test_cascade_status_alt_key_handling(monkeypatch) -> None:
    """Check that alternate environment variable keys are respected."""
    # Pick a provider that defines an alternate key environment variable.
    alt_provider = next((p for p in L._PROVIDERS if p.get("key_env_alt")), None)
    if not alt_provider:
        # Skip test if no provider defines an alternate key.
        return

    # Ensure both primary and alternate keys are absent.
    monkeypatch.delenv(alt_provider["key_env"], raising=False)
    monkeypatch.delenv(alt_provider["key_env_alt"], raising=False)

    # Set only the alternate key.
    monkeypatch.setenv(alt_provider["key_env_alt"], "alt_dummy_key")

    status = L.cascade_status(probe=False)
    providers_status = status["providers"]
    assert providers_status[alt_provider["name"]]["has_key"] is True
    # Primary key should still be reported as missing.
    assert providers_status[alt_provider["name"]]["has_primary_key"] is False
    # Overall health remains false because no probing was performed.
    assert status["healthy"] is False

    # Verify that the working list is still a list even when no providers are active.
    assert isinstance(status["working"], list)