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
# Unit tests for edge conditions
# ==============================
import unittest
from datetime import datetime, timedelta


class TestAlphaDecayTrackerEdgeCases(unittest.TestCase):
    def setUp(self):
        # Fixed seed for reproducibility
        np.random.seed(0)

        # Create a simple datetime index
        self.start = datetime(2022, 1, 1, 0, 0)
        self.periods = 100
        self.index = pd.date_range(self.start, periods=self.periods, freq="H")

    def test_missing_close_column_raises(self):
        """Ensure a ValueError is raised when 'close' column is absent."""
        signals = pd.Series(np.random.choice([-1, 0, 1], size=self.periods), index=self.index)
        prices = pd.DataFrame(np.random.randn(self.periods, 2), index=self.index, columns=["open", "high"])
        tracker = AlphaDecayTracker()
        with self.assertRaises(ValueError):
            tracker.compute_ic_profile(signals, prices, "test_strategy")

    def test_insufficient_horizons_returns_inf_half_life(self):
        """When not enough horizons have enough data, half-life should be infinite and ic_0 zero."""
        # Signals and prices with very short overlap; less than 30 common points for any horizon
        signals = pd.Series(np.random.choice([-1, 0, 1], size=10), index=self.index[:10])
        prices = pd.DataFrame({"close": np.random.randn(10)}, index=self.index[:10])
        tracker = AlphaDecayTracker()
        profile = tracker.compute_ic_profile(signals, prices, "short_data")
        self.assertEqual(profile.ic_0, 0.0)
        self.assertTrue(np.isinf(profile.half_life_hours))
        self.assertEqual(profile.horizons, {})

    def test_scale_confidence_no_decay_returns_base(self):
        """If half-life is infinite, confidence should remain unchanged regardless of staleness."""
        profile = DecayProfile(strategy_name="no_decay", ic_0=0.5, half_life_hours=float("inf"), horizons={})
        tracker = AlphaDecayTracker()
        for staleness in [0, 5, 100]:
            scaled = tracker.scale_confidence(0.8, profile, staleness)
            self.assertAlmostEqual(scaled, 0.8, places=7)

    def test_scale_confidence_negative_staleness(self):
        """Negative staleness should increase confidence but stay bounded by the exponential formula."""
        profile = DecayProfile(strategy_name="neg_stale", ic_0=0.5, half_life_hours=10, horizons={})
        tracker = AlphaDecayTracker()
        base = 0.6
        scaled = tracker.scale_confidence(base, profile, -5)  # negative staleness
        expected_decay = np.exp(5 * np.log(2) / 10)  # exp(+)
        self.assertAlmostEqual(scaled, base * expected_decay, places=7)


if __name__ == "__main__":
    unittest.main()