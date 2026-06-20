"""Tests for the LLM cascade observability: metrics logging + cascade_status probe."""
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / ".github" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import llm_common as L  # noqa: E402


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
