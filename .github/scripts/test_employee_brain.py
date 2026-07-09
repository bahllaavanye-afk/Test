"""Tests for per-employee brains: private memory, recall, and collaboration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import employee_brain as eb


@pytest.fixture()
def state(tmp_path: Path) -> Path:
    p = tmp_path / "agent_memory.json"
    p.write_text(json.dumps({"version": 1, "employee_context": {}, "peer_learnings": []}))
    return p


def test_private_history_is_per_employee(state: Path):
    a = eb.brain_for("vp_eng", state)
    a.remember("task", "reviewed CI flakiness in test_risk_engine.py")
    a.save()
    b = eb.brain_for("risk_eng", state)
    b.remember("task", "flagged 70/30 bucket breach")
    b.save()

    # each employee only sees its own history
    assert len(eb.brain_for("vp_eng", state).recent_history()) == 1
    assert "CI flakiness" in eb.brain_for("vp_eng", state).recent_history()[-1]["text"]
    assert len(eb.brain_for("risk_eng", state).recent_history()) == 1
    assert "bucket breach" in eb.brain_for("risk_eng", state).recent_history()[-1]["text"]


def test_history_survives_reload_and_is_capped(state: Path):
    b = eb.brain_for("ml_lead", state)
    for i in range(100):
        b.remember("note", f"entry {i}")
    b.save()
    reloaded = eb.brain_for("ml_lead", state)
    hist = reloaded.recent_history(1000)
    assert len(hist) == eb._HISTORY_CAP           # capped
    assert hist[-1]["text"] == "entry 99"         # newest kept
    assert hist[0]["text"] == f"entry {100 - eb._HISTORY_CAP}"


def test_durable_facts(state: Path):
    b = eb.brain_for("alpha_dir", state)
    b.note("preferred_lookback", 60)
    b.note("watchlist", ["SPY", "QQQ"])
    b.save()
    r = eb.brain_for("alpha_dir", state)
    assert r.get_fact("preferred_lookback") == 60
    assert r.get_fact("watchlist") == ["SPY", "QQQ"]
    assert r.get_fact("missing", "d") == "d"


def test_collaboration_share_and_learn_from_peers(state: Path):
    r = eb.brain_for("risk_eng", state)
    r.share("drawdown breached on desk-crypto")
    r.save()
    m = eb.brain_for("ml_lead", state)
    m.share("TFT overfitting: val 1.8 vs train 3.2")
    m.save()

    # ml_lead learns from OTHERS (excludes its own posts by default)
    peers = eb.brain_for("ml_lead", state).learn_from_peers(10)
    authors = {p["from"] for p in peers}
    assert "risk_eng" in authors
    assert "ml_lead" not in authors
    assert any("drawdown breached" in p["text"] for p in peers)


def test_learn_from_peers_parses_legacy_string_entries(state: Path):
    # pre-existing bus entries in the legacy "[author @ ts] text" form
    data = json.loads(state.read_text())
    data["peer_learnings"] = ["[standup_agent @ 2026-07-09T10:54:00+00:00] shipped nothing today"]
    state.write_text(json.dumps(data))
    peers = eb.brain_for("vp_eng", state).learn_from_peers(5)
    assert peers[-1]["from"] == "standup_agent"
    assert "shipped nothing" in peers[-1]["text"]


def test_share_keeps_legacy_string_format_for_back_compat(state: Path):
    b = eb.brain_for("vp_eng", state)
    b.share("hello team")
    b.save()
    raw = json.loads(state.read_text())["peer_learnings"][-1]
    assert isinstance(raw, str)
    assert raw.startswith("[vp_eng @ ") and raw.endswith("hello team")


def test_context_block_empty_when_cold(state: Path):
    assert eb.brain_for("new_hire", state).context_block() == ""


def test_context_block_includes_memory_and_peers(state: Path):
    eb.brain_for("alpha_dir", state);
    other = eb.brain_for("risk_eng", state); other.share("VIX regime shifted"); other.save()
    b = eb.brain_for("alpha_dir", state)
    b.note("desk", "equities")
    b.remember("output", "momentum signal on SPY conf 0.82")
    b.save()
    block = eb.brain_for("alpha_dir", state).context_block()
    assert "YOUR MEMORY" in block
    assert "equities" in block            # fact
    assert "momentum signal" in block     # own history
    assert "VIX regime shifted" in block  # peer learning


def test_record_interaction_is_fail_open_on_bad_path(tmp_path: Path):
    # unwritable path → returns False, never raises
    bad = tmp_path / "nope" / "x" / "agent_memory.json"
    # parent dirs will be created; make it truly bad by pointing at a file-as-dir
    f = tmp_path / "afile"; f.write_text("x")
    really_bad = f / "agent_memory.json"
    assert eb.record_interaction("vp_eng", "t", "o", "learned", really_bad) is False


def test_record_interaction_records_and_shares(state: Path):
    assert eb.record_interaction("ml_lead", "analyze experiment", "reduce d_model to 32",
                                 share_line="d_model 64→32 cut overfit", path=state) is True
    b = eb.brain_for("ml_lead", state)
    kinds = {e["kind"] for e in b.recent_history(10)}
    assert {"task", "output"} <= kinds
    # the shared line is visible to a peer
    peers = eb.brain_for("risk_eng", state).learn_from_peers(10)
    assert any("d_model 64" in p["text"] for p in peers)
