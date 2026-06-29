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
