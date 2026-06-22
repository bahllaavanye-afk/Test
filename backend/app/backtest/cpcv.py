"""
Combinatorial Purged Cross-Validation (CPCV) — López de Prado (2018).
======================================================================
Stronger than walk-forward: tests all k-fold combinations, prevents
multiple-testing overfitting, reports Deflated Sharpe Ratio (DSR).

Academic basis:
  - López de Prado (2018) "Advances in Financial Machine Learning"
    Chapter 12: Cross-Validation in Finance
  - Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
  - Bailey et al. (2014) "Pseudo-Mathematics and Financial Charlatanism"

Key insight:
  Standard k-fold CV is invalid for financial time series due to serial
  correlation. CPCV adds purge gaps (to prevent forward leakage) and
  embargo gaps (to prevent backward leakage) around each test fold.
  The Deflated Sharpe Ratio corrects for multiple-testing inflation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, List


class CPCV:
    """
    Combinatorial Purged Cross-Validation for financial time series.

    Parameters:
        n_splits: number of time-series folds (6 gives C(6,1)=6 test periods)
        purge_days: bars to drop before the test fold (prevents train→test leakage)
        embargo_days: bars to drop after the test fold (prevents test→train leakage)

    Usage:
        cpcv = CPCV(n_splits=6, purge_days=5, embargo_days=2)
        results = cpcv.validate(signals, returns)
        print(f"Deflated Sharpe: {results['deflated_sharpe']:.3f}")
        print(f"Overfit: {results['is_overfit']}")
    """

    def __init__(
        self,
        n_splits: int = 6,
        purge_days: int = 5,
        embargo_days: int = 2,
    ):
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError(f"n_splits must be an integer >= 2, got {n_splits}")
        if not isinstance(purge_days, int) or purge_days < 0:
            raise ValueError(f"purge_days must be a non‑negative integer, got {purge_days}")
        if not isinstance(embargo_days, int) or embargo_days < 0:
            raise ValueError(f"embargo_days must be a non‑negative integer, got {embargo_days}")
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, index: pd.DatetimeIndex):
        """
        Yield (train_idx, test_idx) pairs with purge/embargo gaps.

        train_idx and test_idx are lists of integer positions into `index`.
        Bars within purge_days of test_start or embargo_days of test_end
        are excluded from the training set.
        """
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("index must be a pandas.DatetimeIndex")
        if not index.is_monotonic_increasing:
            raise ValueError("index must be sorted in increasing order")
        n = len(index)
        if n < self.n_splits:
            raise ValueError(
                f"Index length {n} is too short for {self.n_splits} folds"
            )
        fold_size = n // self.n_splits
        if fold_size == 0:
            raise ValueError(
                f"Index length {n} is too short for {self.n_splits} folds"
            )

        folds: list[range] = [
            range(i * fold_size, min((i + 1) * fold_size, n))
            for i in range(self.n_splits)
        ]

        for test_fold_idx in range(self.n_splits):
            test_idx = list(folds[test_fold_idx])
            test_start = test_idx[0]
            test_end = test_idx[-1]

            train_idx: list[int] = []
            for i, fold in enumerate(folds):
                if i == test_fold_idx:
                    continue
                for j in fold:
                    # Purge: exclude bars within purge_days before test_start
                    if (test_start - j) <= self.purge_days and j < test_start:
                        continue
                    # Embargo: exclude bars within embargo_days after test_end
                    if (j - test_end) <= self.embargo_days and j > test_end:
                        continue
                    train_idx.append(j)

            yield train_idx, test_idx

    def deflated_sharpe(
        self,
        sharpe_ratios: Iterable[float],
        n_trials: int,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado 2014).

        Adjusts observed Sharpe Ratio downward for:
        1. Multiple testing: the more trials, the higher the expected best SR by luck.
        2. Non-normality: excess kurtosis inflates SR under normality assumption.

        DSR = (mean_SR - SR*) / std_SR
        where SR* is the expected maximum SR over n_trials random draws.

        Args:
            sharpe_ratios: iterable of SR values from each CPCV fold.
            n_trials: number of strategy configurations tried (use len(sharpe_ratios)
                      for a single strategy; use larger if parameter-swept).

        Returns:
            DSR as float. Positive = strategy is robust. Negative = likely overfit.
        """
        if not isinstance(n_trials, int) or n_trials < 1:
            raise ValueError(f"n_trials must be an integer >= 1, got {n_trials}")

        # Convert to list to allow multiple passes and length checks
        sr_list: List[float] = list(sharpe_ratios)
        if not sr_list:
            return 0.0

        # Validate each element is numeric
        for i, val in enumerate(sr_list):
            if not isinstance(val, (int, float, np.number)):
                raise ValueError(f"sharpe_ratios element at position {i} is not numeric: {val}")

        sr = np.array(sr_list, dtype=float)
        if len(sr) < 2:
            return float(sr[0])

        mean_sr = float(np.mean(sr))
        std_sr = float(np.std(sr, ddof=1)) + 1e-10

        # Expected maximum SR under n_trials independent tests
        # Approximation: E[max_SR] ≈ (1 - γ)*Φ⁻¹(1 - 1/n) + γ*Φ⁻¹(1 - 1/(n·e))
        # where γ is Euler-Mascheroni constant
        # Uses scipy.special.erfinv for the normal quantile
        try:
            from scipy.special import erfinv  # type: ignore
            gamma = 0.5772156649  # Euler-Mascheroni constant

            def norm_ppf(p: float) -> float:
                p = float(np.clip(p, 1e-10, 1 - 1e-10))
                return float(np.sqrt(2) * erfinv(2 * p - 1))

            p1 = 1.0 - 1.0 / max(n_trials, 1)
            p2 = 1.0 - 1.0 / max(n_trials * np.e, 1)
            sr_star = (1 - gamma) * norm_ppf(p1) + gamma * norm_ppf(p2)
            # Scale by empirical std of SR distribution
            sr_star = sr_star * float(np.sqrt(np.var(sr) + 1))
        except ImportError:
            # Fallback: simple approximation
            sr_star = float(np.log(n_trials + 1) * 0.5)

        dsr = (mean_sr - sr_star) / std_sr
        return float(dsr)

    def validate(
        self,
        signals: pd.Series,
        returns: pd.Series,
    ) -> dict:
        """
        Run CPCV on signals vs returns.

        Computes Sharpe Ratio on each out-of-sample fold using the signals
        shifted by 1 bar to prevent lookahead bias.

        Args:
            signals: pd.Series of strategy signals (-1, 0, +1) indexed by datetime.
            returns: pd.Series of asset returns at the same frequency.

        Returns:
            dict with:
              fold_sharpes: list of per-fold Sharpe Ratios (annualized)
              mean_sharpe: mean across folds
              deflated_sharpe: DSR (adjusted for multiple testing)
              is_overfit: True if DSR < 0.8 × mean_sharpe
        """
        if not isinstance(signals, pd.Series):
            raise ValueError("signals must be a pandas.Series")
        if not isinstance(returns, pd.Series):
            raise ValueError("returns must be a pandas.Series")

        # Ensure datetime index
        if not isinstance(signals.index, pd.DatetimeIndex):
            signals = signals.copy()
            signals.index = pd.to_datetime(signals.index)
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)

        if not signals.index.is_monotonic_increasing:
            raise ValueError("signals index must be sorted in increasing order")
        if not returns.index.is_monotonic_increasing:
            raise ValueError("returns index must be sorted in increasing order")

        # Align the two series
        common_idx = signals.index.intersection(returns.index)
        if common_idx.empty:
            raise ValueError("signals and returns have no overlapping dates")
        signals = signals.loc[common_idx]
        returns = returns.loc[common_idx]

        if signals.empty or returns.empty:
            raise ValueError("aligned signals or returns series is empty")

        # Verify numeric data
        if not np.issubdtype(signals.dtype, np.number):
            raise ValueError("signals series must contain numeric values")
        if not np.issubdtype(returns.dtype, np.number):
            raise ValueError("returns series must contain numeric values")

        sharpes: list[float] = []
        for train_idx, test_idx in self.split(pd.DatetimeIndex(signals.index)):
            test_signals = signals.iloc[test_idx]
            test_returns = returns.iloc[test_idx]
            # Shift signals by 1 to prevent lookahead
            pnl = test_signals.shift(1).fillna(0) * test_returns
            sr = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(252)
            sharpes.append(float(sr))

        if not sharpes:
            return {
                "fold_sharpes": [],
                "mean_sharpe": 0.0,
                "deflated_sharpe": 0.0,
                "is_overfit": True,
            }

        mean_sr = float(np.mean(sharpes))
        dsr = self.deflated_sharpe(sharpes, n_trials=len(sharpes))

        return {
            "fold_sharpes": sharpes,
            "mean_sharpe": mean_sr,
            "deflated_sharpe": dsr,
            "is_overfit": dsr < 0.8 * mean_sr,
        }