"""Unit tests for the strategy promotion pipeline."""
import pytest
from app.tasks.promotion_criteria import check_criteria, TRANSITION_MAP


# ---------------------------------------------------------------------------
# 1. paper → shadow: all criteria met
# ---------------------------------------------------------------------------

def test_criteria_paper_to_shadow_pass():
    metrics = {
        "days_in_stage": 15,
        "sharpe": 0.7,
        "win_rate": 0.50,
        "max_drawdown": -0.10,
        "num_trades": 25,
    }
    passed, failures = check_criteria(metrics, "paper_to_shadow")
    assert passed is True
    assert failures == []


# ---------------------------------------------------------------------------
# 2. paper → shadow: Sharpe too low
# ---------------------------------------------------------------------------

def test_criteria_paper_to_shadow_fail_sharpe():
    metrics = {
        "days_in_stage": 15,
        "sharpe": 0.3,          # below 0.5 threshold
        "win_rate": 0.50,
        "max_drawdown": -0.10,
        "num_trades": 25,
    }
    passed, failures = check_criteria(metrics, "paper_to_shadow")
    assert passed is False
    assert any("Sharpe" in f for f in failures)


# ---------------------------------------------------------------------------
# 3. shadow → staging: all criteria met
# ---------------------------------------------------------------------------

def test_criteria_shadow_to_staging_pass():
    metrics = {
        "days_in_stage": 31,
        "sharpe": 0.9,
        "win_rate": 0.55,
        "max_drawdown": -0.08,
        "num_trades": 45,
    }
    passed, failures = check_criteria(metrics, "shadow_to_staging")
    assert passed is True
    assert failures == []


# ---------------------------------------------------------------------------
# 4. staging → live: all criteria met
# ---------------------------------------------------------------------------

def test_criteria_staging_to_live_pass():
    metrics = {
        "days_in_stage": 61,
        "sharpe": 1.5,
        "win_rate": 0.55,
        "max_drawdown": -0.07,
        "num_trades": 65,
    }
    passed, failures = check_criteria(metrics, "staging_to_live")
    assert passed is True
    assert failures == []


# ---------------------------------------------------------------------------
# 5. staging → live: drawdown too large
# ---------------------------------------------------------------------------

def test_criteria_staging_to_live_fail_drawdown():
    metrics = {
        "days_in_stage": 61,
        "sharpe": 1.5,
        "win_rate": 0.55,
        "max_drawdown": -0.20,  # below -0.10 threshold
        "num_trades": 65,
    }
    passed, failures = check_criteria(metrics, "staging_to_live")
    assert passed is False
    assert any("drawdown" in f.lower() for f in failures)


# ---------------------------------------------------------------------------
# 6. TRANSITION_MAP covers all three active stages
# ---------------------------------------------------------------------------

def test_transition_map_coverage():
    active_stages = {"paper", "shadow", "staging"}
    assert active_stages.issubset(TRANSITION_MAP.keys()), (
        f"TRANSITION_MAP is missing stages: {active_stages - set(TRANSITION_MAP.keys())}"
    )
    # Each value must be a recognised CRITERIA key
    from app.tasks.promotion_criteria import CRITERIA
    for stage, transition in TRANSITION_MAP.items():
        assert transition in CRITERIA, f"Transition '{transition}' not in CRITERIA"
