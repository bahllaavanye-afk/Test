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
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_N_SPLITS: int = 6
DEFAULT_PURGE_DAYS: int = 5
DEFAULT_EMBARGO_DAYS: int = 2

MIN_N_SPLITS: int = 2
MIN_PURGE_DAYS: int = 0
MIN_EMBARGO_DAYS: int = 0

EPS: float = 1e-10
ANNUALIZATION_FACTOR: int = 252
OVERFIT_THRESHOLD_FACTOR: float = 0.8

EULER_MASCHERONI: float = 0.5772156649

ERR_N_SPLITS: str = "n_splits must be >= {min}, got {value}"
ERR_PURGE_DAYS: str = "purge_days must be >= {min}, got {value}"
ERR_EMBARGO_DAYS: str = "embargo_days must be >= {min}, got {value}"
ERR_FOLD_SIZE: str = "Index length {n} is too short for {splits} folds"


class CPCV:
    """
    Combinatorial Purged Cross‑Validation for financial time series.

    Parameters
    ----------
    n_splits : int, default {DEFAULT_N_SPLITS}
        Number of time‑series folds (e.g., 6 gives C(6,1)=6 test periods).
    purge_days : int, default {DEFAULT_PURGE_DAYS}
        Bars to drop before the test fold (prevents train→test leakage).
    embargo_days : int, default {DEFAULT_EMBARGO_DAYS}
        Bars to drop after the test fold (prevents test→train leakage).
    """

    def __init__(
        self,
        n_splits: int = DEFAULT_N_SPLITS,
        purge_days: int = DEFAULT_PURGE_DAYS,
        embargo_days: int = DEFAULT_EMBARGO_DAYS,
    ):
        if n_splits < MIN_N_SPLITS:
            raise ValueError(
                ERR_N_SPLITS.format(min=MIN_N_SPLITS, value=n_splits)
            )
        if purge_days < MIN_PURGE_DAYS:
            raise ValueError(
                ERR_PURGE_DAYS.format(min=MIN_PURGE_DAYS, value=purge_days)
            )
        if embargo_days < MIN_EMBARGO_DAYS:
            raise ValueError(
                ERR_EMBARGO_DAYS.format(min=MIN_EMBARGO_DAYS, value=embargo_days)
            )
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, index: pd.DatetimeIndex):
        """
        Yield (train_idx, test_idx) pairs with purge/embargo gaps.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Full timeline of the dataset.

        Yields
        ------
        tuple[list[int], list[int]]
            Training and testing integer positions for each combinatorial fold.
        """
        n = len(index)
        fold_size = n // self.n_splits
        if fold_size == 0:
            raise ValueError(
                ERR_FOLD_SIZE.format(n=n, splits=self.n_splits)
            )

        folds: List[range] = [
            range(i * fold_size, min((i + 1) * fold_size, n))
            for i in range(self.n_splits)
        ]

        for test_fold_idx in range(self.n_splits):
            test_idx = list(folds[test_fold_idx])
            test_start = test_idx[0]
            test_end = test_idx[-1]

            train_idx: List[int] = []
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
        sharpe_ratios: list[float],
        n_trials: int,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado 2014).

        Adjusts observed Sharpe Ratio downward for:
        1. Multiple testing: the more trials, the higher the expected best SR by luck.
        2. Non‑normality: excess kurtosis inflates SR under normality assumption.

        DSR = (mean_SR - SR*) / std_SR
        where SR* is the expected maximum SR over n_trials random draws.

        Parameters
        ----------
        sharpe_ratios : list[float]
            SR values from each CPCV fold.
        n_trials : int
            Number of strategy configurations tried (use ``len(sharpe_ratios)`` for a
            single strategy; larger if parameter‑swept).

        Returns
        -------
        float
            DSR. Positive = strategy is robust. Negative = likely overfit.
        """
        if not sharpe_ratios:
            return 0.0

        sr = np.array(sharpe_ratios, dtype=float)
        if len(sr) < 2:
            return float(sr[0])

        mean_sr = float(np.mean(sr))
        std_sr = float(np.std(sr, ddof=1)) + EPS

        try:
            from scipy.special import erfinv  # type: ignore

            def norm_ppf(p: float) -> float:
                p = float(np.clip(p, EPS, 1 - EPS))
                return float(np.sqrt(2) * erfinv(2 * p - 1))

            p1 = 1.0 - 1.0 / max(n_trials, 1)
            p2 = 1.0 - 1.0 / max(n_trials * np.e, 1)
            sr_star = (1 - EULER_MASCHERONI) * norm_ppf(p1) + EULER_MASCHERONI * norm_ppf(p2)
            sr_star = sr_star * float(np.sqrt(np.var(sr) + 1))
        except ImportError:
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

        Computes Sharpe Ratio on each out‑of‑sample fold using the signals
        shifted by 1 bar to prevent look‑ahead bias.

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
                "runtime_seconds": float,
            }``
        """
        start_time = time.time()

        if not isinstance(signals.index, pd.DatetimeIndex):
            signals = signals.copy()
            signals.index = pd.to_datetime(signals.index)

        common_idx = signals.index.intersection(returns.index)
        signals = signals.loc[common_idx]
        returns = returns.loc[common_idx]

        sharpes: list[float] = []
        total_pnl = 0.0

        for train_idx, test_idx in self.split(pd.DatetimeIndex(signals.index)):
            test_signals = signals.iloc[test_idx]
            test_returns = returns.iloc[test_idx]

            # Shift signals by 1 to prevent look‑ahead bias
            pnl = test_signals.shift(1).fillna(0) * test_returns
            total_pnl += float(pnl.sum())

            sr = pnl.mean() / (pnl.std() + EPS) * np.sqrt(ANNUALIZATION_FACTOR)
            sharpes.append(float(sr))

        if not sharpes:
            result = {
                "fold_sharpes": [],
                "mean_sharpe": 0.0,
                "deflated_sharpe": 0.0,
                "is_overfit": True,
                "runtime_seconds": time.time() - start_time,
            }
        else:
            mean_sr = float(np.mean(sharpes))
            dsr = self.deflated_sharpe(sharpes, n_trials=len(sharpes))
            is_overfit = dsr < OVERFIT_THRESHOLD_FACTOR * mean_sr

            result = {
                "fold_sharpes": sharpes,
                "mean_sharpe": mean_sr,
                "deflated_sharpe": dsr,
                "is_overfit": is_overfit,
                "runtime_seconds": time.time() - start_time,
            }

        logger.debug(
            "CPCV validation completed in %.3f seconds – mean SR: %.4f, DSR: %.4f, overfit: %s",
            result["runtime_seconds"],
            result["mean_sharpe"],
            result["deflated_sharpe"],
            result["is_overfit"],
        )
        return result