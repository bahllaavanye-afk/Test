"""Guards the Claude daily budget cap — Claude is used, but never drains credits."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(tmp_budget: Path):
    spec = importlib.util.spec_from_file_location("llmc_under_test", Path(__file__).parent / "llm_common.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["llmc_under_test"] = m
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    m._CLAUDE_BUDGET_FILE = tmp_budget
    return m


def test_budget_starts_empty_and_allows(tmp_path):
    m = _load(tmp_path / "b.json")
    m._CLAUDE_DAILY_BUDGET_USD = 1.0
    assert m._claude_spent_today() == 0.0
    assert m._claude_budget_ok() is True


def test_spend_is_metered_by_token_usage(tmp_path):
    m = _load(tmp_path / "b.json")
    m._CLAUDE_DAILY_BUDGET_USD = 1.0
    # haiku pricing $1/$5 per MTok → 100k in + 50k out = 0.10 + 0.25 = 0.35
    m._record_claude_spend("claude-haiku-4-5", {"input_tokens": 100_000, "output_tokens": 50_000})
    assert abs(m._claude_spent_today() - 0.35) < 1e-9
    assert m._claude_budget_ok() is True


def test_cap_blocks_once_exceeded(tmp_path):
    m = _load(tmp_path / "b.json")
    m._CLAUDE_DAILY_BUDGET_USD = 0.50
    m._record_claude_spend("claude-haiku-4-5", {"input_tokens": 100_000, "output_tokens": 50_000})  # 0.35
    assert m._claude_budget_ok() is True
    m._record_claude_spend("claude-haiku-4-5", {"input_tokens": 100_000, "output_tokens": 50_000})  # 0.70
    assert m._claude_spent_today() > 0.50
    assert m._claude_budget_ok() is False   # Claude now skipped → falls back to free tiers


def test_ledger_resets_on_new_utc_day(tmp_path):
    import json
    p = tmp_path / "b.json"
    m = _load(p)
    m._CLAUDE_DAILY_BUDGET_USD = 0.50
    p.write_text(json.dumps({"date": "2020-01-01", "spent_usd": 99.0}))  # stale day
    assert m._claude_spent_today() == 0.0     # yesterday's spend doesn't count
    assert m._claude_budget_ok() is True


def test_call_claude_skips_when_over_budget(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone
    p = tmp_path / "b.json"
    m = _load(p)
    m._CLAUDE_DAILY_BUDGET_USD = 0.10
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    p.write_text(json.dumps({"date": today, "spent_usd": 5.0}))   # already over
    monkeypatch.setattr(m, "_env_keys", lambda *a: ["sk-test"])   # keys present
    # must return None WITHOUT making any HTTP call
    def _boom(*a, **k):
        raise AssertionError("Claude called despite budget exhausted")
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert m._call_claude("sys", "prompt", 100, 0.0) is None
