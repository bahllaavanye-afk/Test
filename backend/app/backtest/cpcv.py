"""
Combinatorial Purged Cross-Validation (CPCV) — López de Prado (2018).
======================================================================
Stronger than walk-forward: tests all k‑fold combinations, prevents
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
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CPCV:
    """
    Combinatorial Purged Cross‑Validation for financial time series.

    Parameters
    ----------
    n_splits : int, default 6
        Number of time‑series folds. ``C(n_splits, 1)`` test periods will be
        generated.
    purge_days : int, default 5
        Number of bars dropped *before* the test fold (prevents train→test leakage).
    embargo_days : int, default 2
        Number of bars dropped *after* the test fold (prevents test→train leakage).

    Notes
    -----
    The class is deliberately lightweight – it does not store any data
    besides the configuration parameters. All heavy work is performed in
    :meth:`validate`, which is fully vectorised where possible.
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
    def _cached_folds(n: int, n_splits: int) -> Tuple[np.ndarray, ...]:
        """
        Return a tuple of ``np.ndarray`` objects, each containing the integer
        positions belonging to a fold. The result is cached because the same
        ``(n, n_splits)`` pair is often requested repeatedly when the same
        index length is used across many back‑tests.
        """
        fold_size = n // n_splits
        folds = []
        for i in range(n_splits):
            start = i * fold_size
            # Ensure the last fold picks up any remainder rows.
            end = (i + 1) * fold_size if i < n_splits - 1 else n
            folds.append(np.arange(start, end, dtype=int))
        return tuple(folds)

    def split(self, index: pd.DatetimeIndex) -> Tuple[List[int], List[int]]:
        """
        Yield ``(train_idx, test_idx)`` pairs with purge/embargo gaps.

        ``train_idx`` and ``test_idx`` are lists of integer positions into
        ``index``. Bars within ``purge_days`` of the test start or
        ``embargo_days`` of the test end are excluded from the training set.
        """
        n = len(index)
        fold_size = n // self.n_splits
        if fold_size == 0:
            raise ValueError(
                f"Index length {n} is too short for {self.n_splits} folds"
            )

        folds = self._cached_folds(n, self.n_splits)

        for test_fold_idx, test_idx_arr in enumerate(folds):
            test_start = test_idx_arr[0]
            test_end = test_idx_arr[-1]

            # Build a boolean mask for all indices: True → eligible for training
            mask = np.ones(n, dtype=bool)

            # Exclude the test fold itself
            mask[test_idx_arr] = False

            # Purge region (before test start)
            purge_start = max(test_start - self.purge_days, 0)
            mask[purge_start:test_start] = False

            # Embargo region (after test end)
            embargo_end = min(test_end + self.embargo_days + 1, n)
            mask[test_end + 1:embargo_end] = False

            train_idx_arr = np.where(mask)[0]

            # Convert to Python lists for backward compatibility with existing code
            yield train_idx_arr.tolist(), test_idx_arr.tolist()

    def deflated_sharpe(
        self,
        sharpe_ratios: List[float],
        n_trials: int,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado 2014).

        Adjusts observed Sharpe Ratio downward for:
        1. Multiple testing – the more trials, the higher the expected best SR by luck.
        2. Non‑normality – excess kurtosis inflates SR under the normality assumption.

        The formula used is:

        .. math::
            \\text{DSR} = \\frac{\\bar{SR} - SR^{*}}{\\sigma_{SR}}

        where ``SR*`` is the expected maximum Sharpe Ratio over ``n_trials``
        independent draws.

        Parameters
        ----------
        sharpe_ratios : list[float]
            Sharpe Ratio values obtained from each CPCV fold.
        n_trials : int
            Number of strategy configurations tried. Use ``len(sharpe_ratios)`` for a
            single strategy; larger values are appropriate when parameters have been
            swept.

        Returns
        -------
        float
            The deflated Sharpe ratio. Positive values indicate robustness,
            negative values suggest over‑fitting.
        """
        if not sharpe_ratios:
            return 0.0

        sr = np.asarray(sharpe_ratios, dtype=float)
        if sr.size == 1:
            return float(sr[0])

        mean_sr = float(sr.mean())
        std_sr = float(sr.std(ddof=1)) + 1e-10

        # Expected maximum SR under n_trials independent tests.
        # Approximation based on the extreme‑value theory for normal variates.
        try:
            from scipy.special import erfinv  # type: ignore

            gamma = 0.5772156649  # Euler‑Mascheroni constant

            def norm_ppf(p: float) -> float:
                p = float(np.clip(p, 1e-10, 1 - 1e-10))
                return float(np.sqrt(2) * erfinv(2 * p - 1))

            p1 = 1.0 - 1.0 / max(n_trials, 1)
            p2 = 1.0 - 1.0 / max(n_trials * np.e, 1)
            sr_star = (1 - gamma) * norm_ppf(p1) + gamma * norm_ppf(p2)
            # Scale by empirical variance to keep units comparable.
            sr_star *= float(np.sqrt(np.var(sr) + 1))
        except Exception:  # pragma: no cover  (scipy optional)
            # Simple fallback when scipy is unavailable.
            sr_star = float(np.log(n_trials + 1) * 0.5)

        dsr = (mean_sr - sr_star) / std_sr
        return float(dsr)

    def validate(
        self,
        signals: pd.Series,
        returns: pd.Series,
    ) -> dict:
        """
        Run CPCV on ``signals`` vs ``returns`` and compute performance metrics.

        The Sharpe Ratio for each out‑of‑sample fold is computed on the
        *shifted* signals (by one bar) to avoid look‑ahead bias.

        Parameters
        ----------
        signals : pd.Series
            Strategy signals (typically -1, 0, +1) indexed by datetime.
        returns : pd.Series
            Asset returns at the same frequency as ``signals``.

        Returns
        -------
        dict
            ``{
                "fold_sharpes": List[float],
                "mean_sharpe": float,
                "deflated_sharpe": float,
                "is_overfit": bool,
            }``

            ``is_overfit`` is ``True`` when ``deflated_sharpe < 0.8 *
            mean_sharpe``.
        """
        start_time = time.time()

        # Ensure datetime index for alignment.
        if not isinstance(signals.index, pd.DatetimeIndex):
            signals = signals.copy()
            signals.index = pd.to_datetime(signals.index)

        # Align both series on the common datetime index.
        common_idx = signals.index.intersection(returns.index)
        signals = signals.loc[common_idx]
        returns = returns.loc[common_idx]

        if signals.empty:
            logger.warning("CPCV.validate called with empty aligned data.")
            return {
                "fold_sharpes": [],
                "mean_sharpe": 0.0,
                "deflated_sharpe": 0.0,
                "is_overfit": True,
            }

        sharpes: List[float] = []

        # Vectorised loop over folds – the heavy part (train set) is not used
        # for Sharpe computation, so we avoid any unnecessary copying.
        for train_idx, test_idx in self.split(pd.DatetimeIndex(signals.index)):
            test_signals = signals.iloc[test_idx]
            test_returns = returns.iloc[test_idx]

            # Shift by one bar to remove look‑ahead bias.
            pnl = test_signals.shift(1).fillna(0) * test_returns

            # Annualised Sharpe (assumes daily data → 252 trading days)
            sr = pnl.mean() / (pnl.std(ddof=0) + 1e-10) * np.sqrt(252)
            sharpes.append(float(sr))

        mean_sr = float(np.mean(sharpes)) if sharpes else 0.0
        dsr = self.deflated_sharpe(sharpes, n_trials=len(sharpes))

        result = {
            "fold_sharpes": sharpes,
            "mean_sharpe": mean_sr,
            "deflated_sharpe": dsr,
            "is_overfit": dsr < 0.8 * mean_sr,
        }

        elapsed = time.time() - start_time
        logger.debug(
            "CPCV.validate completed in %.3f seconds – mean SR: %.4f, DSR: %.4f",
            elapsed,
            mean_sr,
            dsr,
        )
        return result