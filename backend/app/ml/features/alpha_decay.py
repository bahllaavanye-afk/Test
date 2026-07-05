"""
Measures IC (Information Coefficient) decay for each strategy signal.
IC = Spearman correlation between signal and subsequent forward return.
Fits exponential decay: IC(t) = IC_0 * exp(-lambda * t)

Usage:
    tracker = AlphaDecayTracker()
    profile = tracker.compute_ic_profile(signals, prices, "momentum")
    scaled_conf = tracker.scale_confidence(0.7, profile, staleness_hours=2)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.stats import spearmanr
from scipy.optimize import curve_fit

@dataclass
class DecayProfile:
    strategy_name: str
    ic_0: float           # IC at t=0
    half_life_hours: float  # hours until IC halves
    horizons: dict = field(default_factory=dict)  # {horizon_hours: ic_value}


class AlphaDecayTracker:
    """
    Measures how quickly a strategy's predictive power decays over time.

    Fits an exponential decay model to Spearman IC across multiple horizons.
    Used to scale down signal confidence when signals are stale.
    """

    # Horizons to measure IC at: 1h, 4h, 1d, 5d, 20d
    HORIZONS: list[int] = [1, 4, 24, 120, 480]

    def compute_ic_profile(
        self,
        signals: pd.Series,
        prices: pd.DataFrame,
        strategy_name: str,
    ) -> DecayProfile:
        """
        Compute IC at each horizon and fit exponential decay.

        Args:
            signals: pd.Series of -1/0/+1 indexed by datetime
            prices: pd.DataFrame with 'close' column at same frequency as signals
            strategy_name: name for labelling the profile

        Returns:
            DecayProfile with IC at each horizon and fitted half-life in hours.
            Raises ValueError if prices has no 'close' column.
        """
        if "close" not in prices.columns:
            raise ValueError("prices DataFrame must contain a 'close' column")

        ics: dict[int, float] = {}

        for h in self.HORIZONS:
            fwd_ret = prices["close"].pct_change(h).shift(-h)
            common = signals.index.intersection(fwd_ret.index)
            if len(common) < 30:
                continue

            s = signals.loc[common].dropna()
            r = fwd_ret.loc[s.index].dropna()
            s = s.loc[r.index]

            if len(s) < 20:
                continue

            ic_val, _ = spearmanr(s, r)
            if not np.isnan(ic_val):
                ics[h] = float(ic_val)

        if len(ics) < 2:
            return DecayProfile(
                strategy_name=strategy_name,
                ic_0=0.0,
                half_life_hours=float("inf"),
                horizons=ics,
            )

        horizons_arr = np.array(list(ics.keys()), dtype=float)
        ic_arr = np.array(list(ics.values()), dtype=float)

        try:
            def exp_decay(t: np.ndarray, ic0: float, lam: float) -> np.ndarray:
                return ic0 * np.exp(-lam * t)

            popt, _ = curve_fit(
                exp_decay,
                horizons_arr,
                ic_arr,
                p0=[float(ic_arr[0]), 0.01],
                maxfev=1000,
            )
            ic_0, lam = float(popt[0]), float(popt[1])
            half_life = np.log(2) / lam if lam > 0 else float("inf")
        except Exception:
            ic_0 = float(ic_arr[0]) if len(ic_arr) > 0 else 0.0
            half_life = float("inf")

        return DecayProfile(
            strategy_name=strategy_name,
            ic_0=ic_0,
            half_life_hours=float(half_life),
            horizons=ics,
        )

    def scale_confidence(
        self,
        base_confidence: float,
        profile: DecayProfile,
        staleness_hours: float,
    ) -> float:
        """
        Scale a signal's confidence downward based on how stale it is.

        Args:
            base_confidence: raw confidence score [0, 1]
            profile: fitted DecayProfile for the strategy
            staleness_hours: hours since the signal was generated

        Returns:
            Adjusted confidence in [0, 1].  Returns base_confidence unchanged
            when half-life is infinite (signal does not decay).
        """
        if profile.half_life_hours == float("inf") or profile.half_life_hours <= 0:
            return float(base_confidence)

        decay = np.exp(
            -staleness_hours * np.log(2) / profile.half_life_hours
        )
        return float(base_confidence * max(float(decay), 0.0))


# ==============================
# Unit tests for edge cases
# ==============================
import pytest


def test_compute_ic_profile_insufficient_data():
    """When there are not enough overlapping points, the profile should default to zero IC and infinite half‑life."""
    rng = pd.date_range("2023-01-01", periods=10, freq="H")
    signals = pd.Series(np.random.choice([-1, 0, 1], size=10), index=rng)
    prices = pd.DataFrame({"close": np.random.rand(10)}, index=rng)

    tracker = AlphaDecayTracker()
    profile = tracker.compute_ic_profile(signals, prices, "test_strategy")

    assert profile.ic_0 == 0.0
    assert profile.half_life_hours == float("inf")
    # horizons should be empty because no horizon met the minimum sample size
    assert profile.horizons == {}


def test_compute_ic_profile_missing_close_column():
    """A missing 'close' column must raise a ValueError."""
    rng = pd.date_range("2023-01-01", periods=50, freq="H")
    signals = pd.Series(np.random.choice([-1, 0, 1], size=50), index=rng)
    prices = pd.DataFrame({"open": np.random.rand(50)}, index=rng)  # no close column

    tracker = AlphaDecayTracker()
    with pytest.raises(ValueError, match="prices DataFrame must contain a 'close' column"):
        tracker.compute_ic_profile(signals, prices, "test_strategy")


def test_scale_confidence_boundary_conditions():
    """Check behaviour when half‑life is zero/negative and when staleness is large."""
    # Case 1: zero half‑life should return the base confidence unchanged
    profile_zero = DecayProfile(strategy_name="zero", ic_0=0.5, half_life_hours=0.0, horizons={})
    tracker = AlphaDecayTracker()
    assert tracker.scale_confidence(0.8, profile_zero, staleness_hours=10) == 0.8

    # Case 2: finite half‑life with very large staleness should decay towards zero but never become negative
    profile_finite = DecayProfile(strategy_name="finite", ic_0=0.5, half_life_hours=5.0, horizons={})
    conf = tracker.scale_confidence(0.9, profile_finite, staleness_hours=100)
    assert 0.0 <= conf <= 0.9
    # The decay factor for 100 hours with half‑life 5h is exp(-100*log2/5) ≈ 2^-20 ≈ 9.5e-07
    expected_decay = np.exp(-100 * np.log(2) / 5.0)
    assert np.isclose(conf, 0.9 * expected_decay, atol=1e-12)


# The tests can be run with pytest in the repository root:
#   pytest -q backend/app/ml/features/alpha_decay.py