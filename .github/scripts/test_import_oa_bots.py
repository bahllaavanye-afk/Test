"""Tests for scripts/import_oa_bots.py — OA dump → BOT_TEMPLATES entries."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def imp(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("imp_oa", REPO / "scripts" / "import_oa_bots.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["imp_oa"] = m
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    # sandbox templates.py copy
    t = tmp_path / "templates.py"
    t.write_text(REPO.joinpath("backend/app/bots/templates.py").read_text())
    monkeypatch.setattr(m, "TEMPLATES", t)
    return m


def _spread_bot(name="Test BWB 7 DTE"):
    return {
        "name": name, "symbol": "SPY",
        "legs": [
            {"side": "buy", "option_type": "put", "delta": 0.4, "dte": 7, "ratio": 1},
            {"side": "sell", "option_type": "put", "delta": 0.3, "dte": 7, "ratio": 2},
            {"side": "buy", "option_type": "put", "delta": 0.15, "dte": 7, "ratio": 1},
        ],
        "exits": {"tp_pct": 30, "sl_pct": 100, "time_hours": 120},
        "allocation_pct": 2.5, "source_url": "https://optionsalpha.com/x",
    }


def _run(imp, tmp_path, bots, *args):
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps(bots))
    old = sys.argv
    sys.argv = ["import_oa_bots.py", str(dump), *args]
    try:
        return imp.main()
    finally:
        sys.argv = old


def _load_templates(imp):
    ns: dict = {"__annotations__": {}}
    exec(compile(imp.TEMPLATES.read_text(), "t", "exec"), ns)
    return ns["BOT_TEMPLATES"]


def test_import_adds_valid_options_bot(imp, tmp_path):
    assert _run(imp, tmp_path, [_spread_bot()]) == 0
    t = _load_templates(imp)
    key = [k for k in t if "test_bwb" in k][0]
    e = t[key]
    assert e["market_type"] == "options"
    assert e["action"]["type"] == "open_option_spread"
    assert len(e["action"]["legs"]) == 3
    assert e["action"]["size_pct"] == 2.5
    assert {"type": "no_position"} in e["conditions"]          # guard default
    assert e["trigger"]["interval"] == "1m"                     # OA default
    assert {"type": "take_profit", "value": 30.0} in e["exit_rules"]


def test_duplicate_by_name_is_skipped(imp, tmp_path):
    assert _run(imp, tmp_path, [_spread_bot()]) == 0
    before = len(_load_templates(imp))
    assert _run(imp, tmp_path, [_spread_bot()]) == 0            # same again
    assert len(_load_templates(imp)) == before                  # unchanged


def test_bookmarklet_raw_text_is_skipped_not_crashed(imp, tmp_path):
    bots = [{"name": "Some Bot", "source_url": "https://x", "raw_text": "IV 30 ..."}]
    assert _run(imp, tmp_path, bots) == 0
    assert not any("some_bot" in k for k in _load_templates(imp))


def test_bad_leg_skipped(imp, tmp_path):
    bad = _spread_bot("Bad Legs Bot")
    bad["legs"][0]["side"] = "hold"
    assert _run(imp, tmp_path, [bad]) == 0
    assert not any("bad_legs" in k for k in _load_templates(imp))


def test_dry_run_writes_nothing(imp, tmp_path):
    before = imp.TEMPLATES.read_text()
    assert _run(imp, tmp_path, [_spread_bot("Dry Bot")], "--dry-run") == 0
    assert imp.TEMPLATES.read_text() == before


def test_result_file_stays_importable_and_existing_bots_intact(imp, tmp_path):
    assert _run(imp, tmp_path, [_spread_bot()]) == 0
    t = _load_templates(imp)
    assert "oa_friday_14dte_bwb" in t                           # original 15 intact
    assert len(t) >= 50
