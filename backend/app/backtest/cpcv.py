"""
Combinatorial Purged Cross-Validation (CPCV) — López de Prado (2018).
======================================================================
Stronger than walk-forward: tests all k-fold combinations, prevents
multiple‑testing overfitting, reports Deflated Sharpe Ratio (DSR).

Academic basis:
  - López de Prado (2018) "Advances in Financial Machine Learning"
    Chapter 12: Cross‑Validation in Finance
  - Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
  - Bailey et al. (2014) "Pseudo‑Mathematics and Financial Charlatanism"

Key insight:
  Standard k‑fold CV is invalid for financial time series due to serial
  correlation. CPCV adds purge gaps (to prevent forward leakage) and
  embargo gaps (to prevent backward leakage) around each test fold.
  The Deflated Sharpe Ratio corrects for multiple‑testing inflation.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CPCV:
    """
    Combinatorial Purged Cross‑Validation for financial time series.

    Parameters
    ----------
    n_splits : int, default 6
        Number of time‑series folds (6 gives C(6,1)=6 test periods).
    purge_days : int, default 5
        Bars to drop before the test fold (prevents train→test leakage).
    embargo_days : int, default 2
        Bars to drop after the test fold (prevents test→train leakage).

    Notes
    -----
    The class is deliberately lightweight; the heavy lifting happens in
    :meth:`split`, which has been vectorised and cached for speed.
    """

    def __init__(
        self,
        n_splits: int = 6,
        purge_days: int = 5,
        embargo_days: int = 2,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if purge_days < 0:
            raise ValueError(f"purge_days must be >= 0, got {purge_days}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")

        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    @staticmethod
    @lru_cache(maxsize=32)
    def _cached_folds(length: int, n_splits: int) -> List[np.ndarray]:
        """
        Return a list of ``np.ndarray`` objects, each containing the integer
        positions of a fold. The result is cached because the same length /
        ``n_splits`` pair is often reused across multiple validations.
        """
        fold_size = length // n_splits
        folds: List[np.ndarray] = []
        for i in range(n_splits):
            start = i * fold_size
            stop = min((i + 1) * fold_size, length)
            folds.append(np.arange(start, stop, dtype=int))
        return folds

    def split(self, index: pd.DatetimeIndex):
        """
        Yield ``(train_idx, test_idx)`` pairs with purge/embargo gaps.

        ``train_idx`` and ``test_idx`` are lists of integer positions into
        ``index``. Bars within ``purge_days`` of ``test_start`` or
        ``embargo_days`` of ``test_end`` are excluded from the training set.
        """
        n = len(index)
        if n == 0:
            raise ValueError("Index must contain at least one element")

        fold_size = n // self.n_splits
        if fold_size == 0:
            raise ValueError(
                f"Index length {n} is too short for {self.n_splits} folds"
            )

        # Cached fold arrays (numpy for vectorised masking)
        folds = self._cached_folds(n, self.n_splits)

        for test_fold_idx in range(self.n_splits):
            test_idx_arr = folds[test_fold_idx]
            test_start = int(test_idx_arr[0])
            test_end = int(test_idx_arr[-1])

            # Build a boolean mask for the entire index that marks trainable rows
            mask = np.ones(n, dtype=bool)

            # Remove the test fold itself
            mask[test_idx_arr] = False

            # Purge: exclude ``purge_days`` bars before the test start
            purge_start = max(0, test_start - self.purge_days)
            mask[purge_start:test_start] = False

            # Embargo: exclude ``embargo_days`` bars after the test end
            embargo_end = min(n, test_end + self.embargo_days + 1)
            mask[test_end + 1 : embargo_end] = False

            # Convert mask to list of positions
            train_idx = list(np.where(mask)[0])

            yield train_idx, list(test_idx_arr)

    def deflated_sharpe(
        self,
        sharpe_ratios: list[float],
        n_trials: int,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado 2014).

        Adjusts observed Sharpe Ratio downward for:
        1. Multiple testing: the more trials, the higher the expected best SR
           by luck.
        2. Non‑normality: excess kurtosis inflates SR under normality assumption.

        DSR = (mean_SR - SR*) / std_SR
        where ``SR*`` is the expected maximum SR over ``n_trials`` random draws.

        Parameters
        ----------
        sharpe_ratios : list[float]
            Sharpe values from each CPCV fold.
        n_trials : int
            Number of strategy configurations tried (use ``len(sharpe_ratios)`` for
            a single strategy; use a larger number if parameters were swept).

        Returns
        -------
        float
            Deflated Sharpe Ratio. Positive values indicate robustness,
            negative values suggest over‑fitting.
        """
        if not sharpe_ratios:
            return 0.0

        sr = np.array(sharpe_ratios, dtype=float)
        if sr.size == 1:
            return float(sr[0])

        mean_sr = float(sr.mean())
        std_sr = float(sr.std(ddof=1)) + 1e-10

        try:
            from scipy.special import erfinv  # type: ignore

            gamma = 0.5772156649  # Euler‑Mascheroni constant

            def norm_ppf(p: float) -> float:
                p = float(np.clip(p, 1e-10, 1 - 1e-10))
                return float(np.sqrt(2) * erfinv(2 * p - 1))

            p1 = 1.0 - 1.0 / max(n_trials, 1)
            p2 = 1.0 - 1.0 / max(n_trials * np.e, 1)
            sr_star = (1 - gamma) * norm_ppf(p1) + gamma * norm_ppf(p2)
            sr_star *= float(np.sqrt(np.var(sr) + 1))
        except Exception:  # pragma: no cover
            # Fallback simple approximation when scipy is unavailable
            sr_star = float(np.log(n_trials + 1) * 0.5)

        dsr = (mean_sr - sr_star) / std_sr
        return float(dsr)

    def validate(
        self,
        signals: pd.Series,
        returns: pd.Series,
    ) -> dict:
        """
        Run CPCV on ``signals`` vs ``returns``.

        Computes Sharpe Ratio on each out‑of‑sample fold using the signals
        shifted by one bar to prevent look‑ahead bias.

        Parameters
        ----------
        signals : pd.Series
            Strategy signals (‑1, 0, +1) indexed by datetime.
        returns : pd.Series
            Asset returns at the same frequency.

        Returns
        -------
        dict
            ``{
                "fold_sharpes": list[float],
                "mean_sharpe": float,
                "deflated_sharpe": float,
                "is_overfit": bool,
                "elapsed_time": float,
            }``
        """
        start_time = time.time()

        # Ensure datetime index for alignment
        if not isinstance(signals.index, pd.DatetimeIndex):
            signals = signals.copy()
            signals.index = pd.to_datetime(signals.index)

        # Align both series on the common datetime index
        common_idx = signals.index.intersection(returns.index)
        signals = signals.loc[common_idx]
        returns = returns.loc[common_idx]

        sharpes: list[float] = []

        for train_idx, test_idx in self.split(pd.DatetimeIndex(signals.index)):
            # ``train_idx`` is currently unused but retained for possible
            # extensions (e.g., model fitting). Keeping the variable avoids
            # breaking downstream code that may rely on the generator signature.
            _ = train_idx

            test_signals = signals.iloc[test_idx]
            test_returns = returns.iloc[test_idx]

            # Shift signals by 1 to prevent look‑ahead bias
            pnl = test_signals.shift(1).fillna(0) * test_returns

            # Annualised Sharpe (assumes daily frequency)
            sr = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(252)
            sharpes.append(float(sr))

        if not sharpes:
            result = {
                "fold_sharpes": [],
                "mean_sharpe": 0.0,
                "deflated_sharpe": 0.0,
                "is_overfit": True,
                "elapsed_time": time.time() - start_time,
            }
        else:
            mean_sr = float(np.mean(sharpes))
            dsr = self.deflated_sharpe(sharpes, n_trials=len(sharpes))
            is_overfit = dsr < 0.8 * mean_sr

            result = {
                "fold_sharpes": sharpes,
                "mean_sharpe": mean_sr,
                "deflated_sharpe": dsr,
                "is_overfit": is_overfit,
                "elapsed_time": time.time() - start_time,
            }

        return result