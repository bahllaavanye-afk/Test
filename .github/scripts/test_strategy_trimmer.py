"""Tests for the continuous strategy trimmer's gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strategy_trimmer import evaluate_trim


def test_fresh_strategy_never_trimmed():
    # Awful numbers but only 3 trades → too small a sample to judge.
    trim, reason = evaluate_trim({"trades": 3, "win_rate": 0.0, "avg_return_pct": -5.0,
                                  "total_return_pct": -15.0})
    assert not trim and "insufficient" in reason


def test_winner_kept():
    trim, _ = evaluate_trim({"trades": 50, "win_rate": 0.6, "avg_return_pct": 0.4,
                             "total_return_pct": 20.0})
    assert not trim


def test_bleeding_cumulative_return_trimmed():
    trim, reason = evaluate_trim({"trades": 30, "win_rate": 0.45, "avg_return_pct": -0.2,
                                  "total_return_pct": -8.0})
    assert trim and "cumulative return" in reason


def test_no_edge_low_winrate_negative_expectancy_trimmed():
    trim, reason = evaluate_trim({"trades": 40, "win_rate": 0.30, "avg_return_pct": -0.1,
                                  "total_return_pct": -2.0})
    assert trim and "no edge" in reason


def test_negative_expectancy_trimmed():
    trim, reason = evaluate_trim({"trades": 25, "win_rate": 0.5, "avg_return_pct": -0.7,
                                  "total_return_pct": -1.0})
    assert trim and "negative expectancy" in reason


def test_low_winrate_but_positive_expectancy_kept():
    # Low hit rate is fine if the winners are big (positive avg return).
    trim, _ = evaluate_trim({"trades": 40, "win_rate": 0.30, "avg_return_pct": 0.8,
                             "total_return_pct": 12.0})
    assert not trim


def test_min_trades_boundary():
    stats = {"trades": 10, "win_rate": 0.2, "avg_return_pct": -1.0, "total_return_pct": -10.0}
    trim, _ = evaluate_trim(stats, min_trades=10)
    assert trim  # exactly at the sample floor → now judged (and fails)
    trim2, _ = evaluate_trim(stats, min_trades=11)
    assert not trim2  # one short → not judged


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


# ── the trimmer read the ENVELOPE, not the payload ───────────────────────────
# All seven tests above call evaluate_trim() directly and passed throughout.
# But run() — the thing the workflow actually invokes — could never trim
# anything, because load_perf() returned fill_tracker's whole document:
#
#   {"generated_at": ..., "period_days": 30, "strategies": {...},
#    "tracked_order_ids": [...]}
#
# so `for name, stats in perf.items()` iterated the ENVELOPE. Only "strategies"
# is dict-valued, and evaluate_trim() on that blob sees .get("trades") == 0 —
# "insufficient sample". No level of bad performance could ever trigger a trim.
#
# Invisible until the artifact existed: with no perf file, load_perf() returned
# {} and the loop did nothing either way. strategy_auto_tuner.py reads the same
# file and always unwrapped it correctly (`saved.get("strategies", {})`) — the
# two consumers simply disagreed about the schema, and only one was exercised.
#
# Found by running run() against the REAL committed artifact in a sandbox
# rather than by reading the code.

import json
import shutil
import tempfile

import pytest

import strategy_trimmer as _trimmer

_REPO = Path(__file__).resolve().parents[2]
_REAL_PERF = _REPO / "backend" / "performance_log" / "strategy_performance.json"

_ENVELOPE = {
    "generated_at": "2026-07-29T10:19:16Z",
    "period_days": 30,
    "strategies": {
        "avellaneda": {"trades": 10, "win_rate": 0.6,
                       "avg_return_pct": -0.7916, "total_return_pct": -7.9157},
    },
    "tracked_order_ids": ["a", "b"],
}


def _run_against(payload: dict, monkeypatch):
    """Drive the real run() with both files redirected into a sandbox."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "state").mkdir()
    perf = tmp / "perf.json"
    perf.write_text(json.dumps(payload))
    monkeypatch.setattr(_trimmer, "PERF_FILE", perf)
    monkeypatch.setattr(_trimmer, "TRIMS_FILE", tmp / "state" / "trims.json")
    monkeypatch.setattr(_trimmer, "STATE_DIR", tmp / "state")
    events = _trimmer.run()
    written = json.loads((tmp / "state" / "trims.json").read_text())
    return events, written


def test_load_perf_unwraps_the_envelope(monkeypatch, tmp_path):
    f = tmp_path / "perf.json"
    f.write_text(json.dumps(_ENVELOPE))
    monkeypatch.setattr(_trimmer, "PERF_FILE", f)
    got = _trimmer.load_perf()
    assert set(got) == {"avellaneda"}, (
        f"load_perf returned the envelope keys {sorted(got)} instead of the "
        f"per-strategy stats"
    )


def test_run_actually_trims_a_bleeding_strategy(monkeypatch):
    """The regression: evaluate_trim said True while run() produced nothing."""
    events, written = _run_against(_ENVELOPE, monkeypatch)
    assert [e["name"] for e in events] == ["avellaneda"], events
    assert "avellaneda" in written
    assert "cumulative return" in written["avellaneda"]["reason"]


def test_run_leaves_a_healthy_strategy_alone(monkeypatch):
    payload = dict(_ENVELOPE, strategies={
        "winner": {"trades": 50, "win_rate": 0.6,
                   "avg_return_pct": 0.4, "total_return_pct": 20.0}})
    events, written = _run_against(payload, monkeypatch)
    assert events == [] and written == {}


def test_envelope_keys_are_never_treated_as_strategies(monkeypatch):
    """`generated_at` / `period_days` must not reach evaluate_trim."""
    events, written = _run_against(_ENVELOPE, monkeypatch)
    for bad in ("generated_at", "period_days", "tracked_order_ids", "strategies"):
        assert bad not in written, f"envelope key {bad!r} was trimmed as a strategy"


def test_a_bare_map_without_an_envelope_yields_nothing(monkeypatch):
    """Fail-soft: an unexpected shape must not crash or invent trims."""
    events, written = _run_against({"avellaneda": {"trades": 10}}, monkeypatch)
    assert events == [] and written == {}


def test_the_two_consumers_agree_on_the_schema():
    """strategy_auto_tuner.py already unwrapped correctly; they must not drift."""
    tuner = (Path(__file__).parent / "strategy_auto_tuner.py").read_text()
    assert 'get("strategies"' in tuner, "auto-tuner no longer unwraps the envelope"
    trimmer_src = (Path(__file__).parent / "strategy_trimmer.py").read_text()
    assert 'get("strategies"' in trimmer_src, "trimmer no longer unwraps the envelope"


@pytest.mark.skipif(not _REAL_PERF.exists(), reason="artifact not committed yet")
def test_against_the_real_committed_artifact(monkeypatch):
    """The substrate that actually matters. A synthetic envelope can drift from
    what fill_tracker really writes; this cannot."""
    real = json.loads(_REAL_PERF.read_text())
    assert "strategies" in real, "the real artifact no longer has a strategies key"
    events, written = _run_against(real, monkeypatch)
    # every trimmed name must be a real strategy key, never an envelope key
    assert set(written) <= set(real["strategies"]), (
        f"trimmed something that is not a strategy: {set(written) - set(real['strategies'])}"
    )
