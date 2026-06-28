"""Tests for the multi-criteria strategy promotion gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strategy_gate import deflated_sharpe_ratio, passes_promotion_gate


def _good():
    return {
        "test_sharpe": 1.8, "val_sharpe": 1.5, "max_dd": 8.0, "num_trades": 120,
        "sortino": 2.2, "calmar": 1.1, "win_rate": 0.55, "profit_factor": 1.6,
    }


def test_strong_strategy_passes_with_few_trials():
    passed, sc = passes_promotion_gate(_good(), n_trials=5, sharpe_variance_across_trials=0.2)
    assert passed, [k for k, c in sc.items() if not c["ok"]]


def test_low_sharpe_rejected():
    m = _good(); m["test_sharpe"] = 0.4
    passed, sc = passes_promotion_gate(m, n_trials=5)
    assert not passed and not sc["test_sharpe"]["ok"]


def test_too_few_trades_rejected():
    m = _good(); m["num_trades"] = 3
    passed, sc = passes_promotion_gate(m, n_trials=5)
    assert not passed and not sc["num_trades"]["ok"]


def test_overfit_rejected_on_oos_consistency():
    # Great validation Sharpe but test collapses → overfit
    m = _good(); m["val_sharpe"] = 3.0; m["test_sharpe"] = 1.1
    passed, sc = passes_promotion_gate(m, n_trials=5)
    assert not sc["oos_consistency"]["ok"]


def test_bad_profit_factor_rejected_when_present():
    m = _good(); m["profit_factor"] = 0.9
    passed, sc = passes_promotion_gate(m, n_trials=5)
    assert not passed and not sc["profit_factor"]["ok"]


def test_missing_soft_metrics_do_not_fail():
    m = {"test_sharpe": 1.8, "val_sharpe": 1.5, "max_dd": 8.0, "num_trades": 120}
    passed, sc = passes_promotion_gate(m, n_trials=5, sharpe_variance_across_trials=0.2)
    assert passed
    assert sc["sortino"]["ok"] and sc["sortino"]["value"] is None


def test_deflated_sharpe_drops_as_trials_rise():
    # Same Sharpe, more trials tried → lower confidence it's real
    few = deflated_sharpe_ratio(1.5, n_obs=200, n_trials=5, sharpe_variance_across_trials=0.25)
    many = deflated_sharpe_ratio(1.5, n_obs=200, n_trials=2000, sharpe_variance_across_trials=0.25)
    assert few > many
    assert 0.0 <= many <= few <= 1.0


def test_many_trials_haircut_can_reject_a_marginal_sharpe():
    # A marginal Sharpe that would pass at 1 trial gets deflated away at 5000 trials
    m = {"test_sharpe": 1.05, "val_sharpe": 0.9, "max_dd": 10.0, "num_trades": 60}
    passed_few, _ = passes_promotion_gate(m, n_trials=2, sharpe_variance_across_trials=0.5)
    passed_many, sc = passes_promotion_gate(m, n_trials=5000, sharpe_variance_across_trials=0.5)
    assert passed_few and not passed_many
    assert not sc["deflated_sharpe"]["ok"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
