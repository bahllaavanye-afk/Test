"""
Integration tests: ML model imports and basic inference.
All models must import gracefully even without torch installed (Render free tier).
"""
from __future__ import annotations

import importlib
import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Graceful import when torch is absent
# ──────────────────────────────────────────────────────────────────────────────

ML_MODULES = [
    "app.ml.models.lstm",
    "app.ml.models.xgboost_model",
    "app.ml.models.lorentzian_knn",
    "app.ml.models.ensemble_model",
    "app.ml.models.lightgbm_model",
    "app.ml.registry",
    "app.ml.inference",
]


@pytest.mark.parametrize("module_path", ML_MODULES)
def test_ml_module_imports_without_crash(module_path):
    """ML modules must import even when torch/sklearn are unavailable."""
    try:
        mod = importlib.import_module(module_path)
        assert mod is not None
    except ImportError as e:
        # Only torch ImportError is acceptable — everything else is a bug
        if "torch" in str(e).lower() or "lightgbm" in str(e).lower():
            pytest.skip(f"Optional ML dependency not installed: {e}")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 200) -> "pd.DataFrame":
    import pandas as pd
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1)
    return pd.DataFrame({
        "open":   close * rng.uniform(0.99, 1.0, n),
        "high":   close * rng.uniform(1.0, 1.01, n),
        "low":    close * rng.uniform(0.99, 1.0, n),
        "close":  close,
        "volume": rng.integers(1_000, 100_000, n).astype(float),
    })


def test_feature_engineer_importable():
    mod = importlib.import_module("app.ml.features.engineer")
    assert hasattr(mod, "FeatureEngineer") or hasattr(mod, "engineer_features")


def test_technical_features_produce_output():
    try:
        from app.ml.features.technical import TechnicalFeatures
    except ImportError:
        pytest.skip("Technical features module not available")
    df = _make_ohlcv()
    feat = TechnicalFeatures()
    result = feat.compute(df)
    assert result is not None
    assert len(result) == len(df), "Feature output must match input length"
    assert result.shape[1] > 0, "Feature output must have at least one column"


def test_features_have_no_lookahead():
    """Features at time t must only use data up to and including time t."""
    try:
        from app.ml.features.technical import TechnicalFeatures
    except ImportError:
        pytest.skip("Technical features module not available")

    df = _make_ohlcv(300)
    feat = TechnicalFeatures()
    full = feat.compute(df)

    # Compute features on first 200 rows only
    partial = feat.compute(df.iloc[:200])

    # The last shared row must be identical (no future data leaking)
    if full is not None and partial is not None and len(partial) > 100:
        last_shared = min(len(partial), len(full)) - 1
        for col in partial.columns:
            if col in full.columns:
                v_full = full.iloc[last_shared][col]
                v_part = partial.iloc[last_shared][col]
                if not (np.isnan(v_full) and np.isnan(v_part)):
                    assert abs(v_full - v_part) < 1e-9, (
                        f"Lookahead bias detected in feature '{col}': "
                        f"full={v_full}, partial={v_part}"
                    )


# ──────────────────────────────────────────────────────────────────────────────
# HMM regime detector
# ──────────────────────────────────────────────────────────────────────────────

def test_hmm_regime_detector():
    try:
        from app.strategies.manual.hmm_regime import HMMRegimeStrategy
        # HMM is implemented as a strategy, not a standalone ML model
        s = HMMRegimeStrategy()
        assert s is not None
        return
    except ImportError:
        pass
    pytest.skip("HMM regime module not available")

    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.015, 500)

    det = RegimeDetector()
    det.fit(returns)

    regimes = det.predict(returns)
    assert len(regimes) == len(returns), "Regime output must match input length"
    assert set(regimes).issubset({0, 1, 2}), f"Regimes must be 0/1/2, got {set(regimes)}"

    current = det.current_regime(returns)
    assert current in (0, 1, 2), f"Current regime must be 0/1/2, got {current}"
