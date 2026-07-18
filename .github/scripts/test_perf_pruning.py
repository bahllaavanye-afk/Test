"""Performance attribution auto-pruning: proven losers stop trading.

The self-scaling loop sized losers at 0.6x forever — a strategy with 50 trades
and deeply negative Sharpe kept bleeding at reduced size. Pruning gives
weight 0.0 (desk skips it entirely) once evidence is strong: ≥20 trades AND
negative P&L AND sharpe < -0.5. Revival is automatic — weights re-fetch from
the live leaderboard every run.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_MOD = Path(__file__).parent / "desk_order_placer.py"


def _load():
    spec = importlib.util.spec_from_file_location("dop_prune_test", _MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


dop = _load()


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self._b


def _mock_leaderboard(monkeypatch, strategies):
    calls = {"n": 0}
    def fake(req, timeout=15):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp({"access_token": "tok"})
        return _Resp({"strategies": strategies})
    monkeypatch.setattr("urllib.request.urlopen", fake)


def test_proven_loser_is_pruned_to_zero(monkeypatch, capsys):
    _mock_leaderboard(monkeypatch, [
        {"strategy": "bleeder", "trades": 25, "total_pnl": -800.0, "pnl_sharpe": -1.2},
    ])
    w = dop._fetch_performance_weights()
    assert w["bleeder"] == 0.0
    assert "Auto-pruned" in capsys.readouterr().out


def test_young_loser_keeps_reduced_weight_not_pruned(monkeypatch):
    # under the trade-count evidence bar → 0.6x, still trading
    _mock_leaderboard(monkeypatch, [
        {"strategy": "young", "trades": 8, "total_pnl": -300.0, "pnl_sharpe": -2.0},
    ])
    assert dop._fetch_performance_weights()["young"] == dop._WEIGHT_MIN


def test_mild_loser_with_ok_sharpe_not_pruned(monkeypatch):
    # negative P&L but sharpe above the bar → 0.6x, still trading
    _mock_leaderboard(monkeypatch, [
        {"strategy": "mild", "trades": 40, "total_pnl": -50.0, "pnl_sharpe": -0.2},
    ])
    assert dop._fetch_performance_weights()["mild"] == dop._WEIGHT_MIN


def test_winner_still_sizes_up(monkeypatch):
    _mock_leaderboard(monkeypatch, [
        {"strategy": "winner", "trades": 30, "total_pnl": 900.0, "pnl_sharpe": 1.5},
    ])
    w = dop._fetch_performance_weights()["winner"]
    assert 1.0 < w <= dop._WEIGHT_MAX


def test_missing_sharpe_never_prunes(monkeypatch):
    # sharpe None (not yet computed) must NOT count as adverse evidence
    _mock_leaderboard(monkeypatch, [
        {"strategy": "nosharpe", "trades": 50, "total_pnl": -500.0, "pnl_sharpe": None},
    ])
    assert dop._fetch_performance_weights()["nosharpe"] == dop._WEIGHT_MIN


def test_hitrate_rule_prunes_large_sample_coin_toss(monkeypatch):
    # 100+ trades, losing, sub-45% accuracy → pruned even with mild sharpe
    _mock_leaderboard(monkeypatch, [
        {"strategy": "coin_toss", "trades": 120, "total_pnl": -200.0,
         "pnl_sharpe": -0.1, "win_rate": 0.41},
    ])
    assert dop._fetch_performance_weights()["coin_toss"] == 0.0


def test_hitrate_rule_needs_big_sample_and_losses(monkeypatch):
    _mock_leaderboard(monkeypatch, [
        # low win rate but small sample → not pruned
        {"strategy": "small_n", "trades": 60, "total_pnl": -100.0,
         "pnl_sharpe": -0.1, "win_rate": 0.40},
        # low win rate but PROFITABLE (big winners, small losers) → never pruned
        {"strategy": "trend_rider", "trades": 150, "total_pnl": 900.0,
         "pnl_sharpe": 0.8, "win_rate": 0.38},
    ])
    w = dop._fetch_performance_weights()
    assert w["small_n"] == dop._WEIGHT_MIN
    assert w["trend_rider"] > 1.0
