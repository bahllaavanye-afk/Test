"""Tests for the LLM cascade observability: metrics logging + cascade_status probe."""
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import llm_common as L  # noqa: E402


def test_record_metric_appends_jsonl(tmp_path, monkeypatch):
    """Verify that _record_metric writes a well‑formed JSON line for each call."""
    f = tmp_path / "m.jsonl"
    monkeypatch.setattr(L, "_METRICS_FILE", f)

    L._record_metric("groq", True, 123)
    L._record_metric("none", False, 50, "all providers failed")

    lines = f.read_text().strip().splitlines()
    assert len(lines) == 2, "Two metric records should be written"

    first = json.loads(lines[0])
    assert (
        first["provider"] == "groq"
        and first["ok"] is True
        and first["ms"] == 123
    ), "First record fields mismatch"

    second = json.loads(lines[1])
    assert second["ok"] is False, "Second record ok flag should be False"
    assert second["msg"] == "all providers failed", "Error message should be preserved"


def test_record_metric_never_raises(monkeypatch):
    """Ensures that an unwritable metrics path does not raise an exception."""
    # Point to a directory that does not exist; the function must swallow any IOError.
    monkeypatch.setattr(L, "_METRICS_FILE", Path("/nonexistent-dir/xyz/m.jsonl"))
    L._record_metric("groq", True, 1)  # No exception means pass


def test_cascade_status_reports_key_presence(monkeypatch):
    """When only a single provider key is set, cascade_status should reflect that."""
    # Remove all environment variables for providers
    for p in L._PROVIDERS:
        monkeypatch.delenv(p["key_env"], raising=False)
        if p.get("key_env_alt"):
            monkeypatch.delenv(p["key_env_alt"], raising=False)

    # Set the key for the first provider only
    first = L._PROVIDERS[0]
    monkeypatch.setenv(first["key_env"], "k")

    st = L.cascade_status(probe=False)  # probe=False → no network calls
    assert st["providers"][first["name"]]["has_key"] is True, "First provider key should be detected"
    # All other providers must have has_key == False
    for name, info in st["providers"].items():
        if name != first["name"]:
            assert info["has_key"] is False, f"Provider {name} should report no key"
    assert st["healthy"] is False, "System cannot be healthy without any working provider"
    assert isinstance(st["working"], list), "working field must be a list"


def test_cascade_status_all_keys_missing(monkeypatch):
    """When no provider keys are present, cascade_status should indicate no keys and empty working list."""
    for p in L._PROVIDERS:
        monkeypatch.delenv(p["key_env"], raising=False)
        if p.get("key_env_alt"):
            monkeypatch.delenv(p["key_env_alt"], raising=False)

    st = L.cascade_status(probe=False)
    # All providers should report has_key == False
    for info in st["providers"].values():
        assert info["has_key"] is False
    assert st["healthy"] is False
    assert st["working"] == [], "working list should be empty when no keys are set"


def test_cascade_status_key_alt(monkeypatch):
    """Providers that define an alternate environment variable should be recognized."""
    # Identify a provider with an alternate key variable
    alt_provider = next((p for p in L._PROVIDERS if p.get("key_env_alt")), None)
    if not alt_provider:
        # If no provider defines an alternate key, the test is not applicable.
        return

    # Ensure primary key is absent and alternate is present
    monkeypatch.delenv(alt_provider["key_env"], raising=False)
    monkeypatch.setenv(alt_provider["key_env_alt"], "alt_key")

    st = L.cascade_status(probe=False)
    assert (
        st["providers"][alt_provider["name"]]["has_key"] is True
    ), "Alternate key environment variable should be detected"
    # No other provider should be marked as having a key
    for name, info in st["providers"].items():
        if name != alt_provider["name"]:
            assert info["has_key"] is False
    assert st["healthy"] is False
    assert isinstance(st["working"], list)