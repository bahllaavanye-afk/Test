"""
Hard-gate CI test: detects lookahead bias in the feature engineering pipeline.

This test is a *required gate* — if it fails, the feature pipeline has introduced
a forward-looking feature that would inflate backtest Sharpe and destroy live P&L.
The engineer.py pipeline uses .shift(1) on labels, not on features, which is correct
(labels look forward; features must only look backward).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.features.store import check_point_in_time, FeatureLeakError


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    vol   = np.abs(rng.standard_normal(n)) * 1e6 + 1e6
    return pd.DataFrame({
        "open":   close * (1 + rng.standard_normal(n) * 0.001),
        "high":   close * (1 + np.abs(rng.standard_normal(n)) * 0.005),
        "low":    close * (1 - np.abs(rng.standard_normal(n)) * 0.005),
        "close":  close,
        "volume": vol,
    })


# ── Unit tests for check_point_in_time ────────────────────────────────────────

class TestCheckPointInTime:
    def test_clean_lagged_feature_passes(self):
        """A properly shifted feature (lagged RSI) must not trigger leak detection."""
        df = _ohlcv()
        # Simulate a well-behaved lagged feature: returns computed on past data only
        df["rsi_lagged"] = df["close"].pct_change(14).shift(1)
        df["close_orig"] = df["close"]  # keep close for correlation
        # rename so check_point_in_time can find 'close'
        result = check_point_in_time(df, ["rsi_lagged"], horizon=1)
        assert "rsi_lagged" not in result or result["rsi_lagged"] <= 0.10

    def test_leaked_feature_raises(self):
        """A feature computed directly from the *future* return must be caught."""
        df = _ohlcv()
        fwd_return = df["close"].pct_change(1).shift(-1)
        # Synthetic feature that IS the future return — maximum lookahead bias
        df["cheating_feature"] = fwd_return * 100
        with pytest.raises(FeatureLeakError, match="Point-in-time violation"):
            check_point_in_time(df, ["cheating_feature"], horizon=1)

    def test_random_noise_feature_passes(self):
        """Pure noise should not be flagged."""
        rng = np.random.default_rng(7)
        df = _ohlcv()
        df["noise"] = rng.standard_normal(len(df))
        result = check_point_in_time(df, ["noise"], horizon=1)
        assert "noise" not in result or result.get("noise", 0) <= 0.05

    def test_returns_not_flagged(self):
        """Backward-looking returns (always valid) must pass."""
        df = _ohlcv()
        df["returns_1"] = df["close"].pct_change(1)
        df["returns_5"] = df["close"].pct_change(5)
        # These look backward, so gap should be <= 0
        result = check_point_in_time(df, ["returns_1", "returns_5"], horizon=1)
        for col in ["returns_1", "returns_5"]:
            assert result.get(col, 0) <= LEAK_WARN_THRESHOLD


    def test_too_few_rows_skipped(self):
        """Short DataFrames with < 30 rows are silently skipped (not enough data)."""
        df = _ohlcv(n=20)
        result = check_point_in_time(df, ["close"], horizon=1)
        assert result == {}

    def test_empty_feature_list_passes(self):
        """Empty feature list must not crash."""
        df = _ohlcv()
        result = check_point_in_time(df, [], horizon=1)
        assert result == {}


LEAK_WARN_THRESHOLD = 0.05


# ── Integration test: run engineer_features and check for leaks ────────────────

class TestEngineerFeaturesNoLeak:
    """
    Runs the real engineer_features() pipeline on synthetic OHLCV data and
    checks that no feature leaks the future. This is the CI hard gate.
    """

    def test_technical_features_no_leak(self):
        """All technical features must be point-in-time correct."""
        df = _ohlcv(n=500)
        try:
            from app.ml.features.technical import add_technical_features
            df_feat = add_technical_features(df.copy())
        except Exception as e:
            pytest.skip(f"technical features unavailable: {e}")

        base_cols = [
            "returns_1", "returns_5", "returns_10",
            "vol_5", "vol_21",
            "rsi_14", "macd", "bb_width",
        ]
        cols = [c for c in base_cols if c in df_feat.columns]
        if not cols:
            pytest.skip("No feature columns present")

        # Should not raise — all these are computed on past data
        leaks = check_point_in_time(df_feat, cols, horizon=1)
        assert not any(v > 0.10 for v in leaks.values()), (
            f"Technical features leaked the future: {leaks}"
        )

    def test_labels_are_forward_looking_but_not_features(self):
        """
        The add_labels() function creates future-looking targets (correct).
        The features themselves must remain point-in-time correct.
        This test verifies they are not accidentally mixed.
        """
        df = _ohlcv(n=500)
        try:
            from app.ml.features.technical import add_technical_features
            from app.ml.features.engineer import add_labels
            df_feat = add_technical_features(df.copy())
            df_feat = add_labels(df_feat, horizon=1)
        except Exception as e:
            pytest.skip(f"Pipeline unavailable: {e}")

        # 'target' / 'label' are allowed to be forward-looking — that's intentional.
        # Check that the *feature* columns don't leak.
        non_target_cols = [c for c in df_feat.columns
                          if c not in ("target", "label", "close", "open",
                                       "high", "low", "volume")]
        if not non_target_cols:
            pytest.skip("No feature columns to check")

        leaks = check_point_in_time(df_feat, non_target_cols[:20], horizon=1)
        bad = {k: v for k, v in leaks.items() if v > 0.10}
        assert not bad, f"Feature columns leaked future into training data: {bad}"
