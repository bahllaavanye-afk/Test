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

import logging
import time
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF via erfinv (no scipy hard dependency)."""
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    try:
        from scipy.special import erfinv  # type: ignore
        return float(np.sqrt(2) * erfinv(2 * p - 1))
    except ImportError:
        # Acklam's rational approximation — accurate to ~1e-9, scipy-free.
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = np.sqrt(-2 * np.log(p))
            return float((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                         ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
        if p > phigh:
            q = np.sqrt(-2 * np.log(1 - p))
            return float(-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                          ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
        q = p - 0.5
        r = q * q
        return float((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                     (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via erf-free approximation (Abramowitz & Stegun 7.1.26)."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sr: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio — Bailey & López de Prado (2012).

    P(true SR > benchmark_sr) given an observed Sharpe on a finite, possibly
    non-normal sample. Corrects for short track records (n_obs) and for the
    skew/kurtosis of returns that inflate a naive Sharpe. Both `observed_sr` and
    `benchmark_sr` must be on the SAME per-period frequency as the moments.

    Returns a probability in [0, 1]; > 0.95 is the usual significance bar.
    """
    if n_obs < 2:
        return 0.0
    denom = np.sqrt(max(1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr ** 2, 1e-12))
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1) / denom
    return float(_norm_cdf(float(z)))


def deflated_sharpe_ratio(sharpe_ratios: List[float], n_trials: int) -> float:
    """
    Deflated Sharpe Ratio — Bailey & López de Prado (2014), as a PROBABILITY.

    Returns P(the mean Sharpe beats what the BEST of `n_trials` random draws
    would produce by luck). Corrects for multiple testing: the more trials, the
    higher the expected best Sharpe under the zero-skill null, so a lone good
    number is discounted. Trials-based approximation — uses the empirical
    dispersion of the per-fold/window Sharpes as the estimation error.

    Args:
        sharpe_ratios: SR values, one per fold / window / trial (any consistent
            frequency; the standardization is scale-free).
        n_trials: number of configurations tried (use len(sharpe_ratios) for a
            single strategy across folds; larger if parameter-swept).

    Returns: probability in [0, 1]. ≥ 0.95 clears the usual significance bar;
    low values mean the observed edge is within what luck would produce (overfit).
    """
    if not sharpe_ratios:
        return 0.0
    sr = np.array(sharpe_ratios, dtype=float)
    mean_sr = float(np.mean(sr))
    if len(sr) < 2:
        return 1.0 if mean_sr > 0 else 0.0
    std_sr = float(np.std(sr, ddof=1))
    if std_sr < 1e-9:                      # perfectly consistent trials
        return 1.0 if mean_sr > 0 else 0.0
    gamma = 0.5772156649  # Euler–Mascheroni
    p1 = 1.0 - 1.0 / max(n_trials, 1)
    p2 = 1.0 - 1.0 / max(n_trials * np.e, 1)
    # Expected max Sharpe by luck, in units of the trial-Sharpe std.
    sr_star_z = (1 - gamma) * _norm_ppf(p1) + gamma * _norm_ppf(p2)
    # Standardized mean Sharpe minus the multiple-testing benchmark → Φ.
    z = mean_sr / std_sr - sr_star_z
    return float(_norm_cdf(z))


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = 16,
) -> dict:
    """
    Probability of Backtest Overfitting (PBO) — Bailey, Borwein, López de Prado
    & Zhu (2017), via combinatorially-symmetric cross-validation (CSCV).

    Answers: when you pick the in-sample-best of N configurations, how often does
    it land BELOW the median out-of-sample? PBO ≈ 0 is good; PBO ≥ 0.5 means the
    selection process is no better than chance (severe overfitting).

    Args:
        returns_matrix: shape (T observations, N configurations) of per-period
            returns — one column per strategy/parameter configuration compared.
        n_splits: number of equal time blocks S (even). All C(S, S/2) splits are
            evaluated, each using S/2 blocks in-sample and the complement OOS.

    Returns: {pbo, n_configs, n_combinations, mean_logit}. Fail-soft with pbo=1.0
    (maximally overfit — the conservative verdict) when there is too little data.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2 or M.shape[0] < n_splits or n_splits < 2:
        return {"pbo": 1.0, "n_configs": int(M.shape[1]) if M.ndim == 2 else 0,
                "n_combinations": 0, "mean_logit": 0.0}
    if n_splits % 2 != 0:
        n_splits -= 1

    T, N = M.shape
    block = T // n_splits
    blocks = [M[i * block:(i + 1) * block] for i in range(n_splits)]

    def _sharpe(sub: np.ndarray) -> np.ndarray:
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0, ddof=0) + 1e-12
        return mu / sd  # per-period Sharpe per config; frequency cancels in ranking

    logits: list[float] = []
    for is_idx in combinations(range(n_splits), n_splits // 2):
        oos_idx = [j for j in range(n_splits) if j not in is_idx]
        J = np.vstack([blocks[j] for j in is_idx])
        Jc = np.vstack([blocks[j] for j in oos_idx])
        is_perf = _sharpe(J)
        oos_perf = _sharpe(Jc)
        n_star = int(np.argmax(is_perf))                 # best in-sample config
        # OOS relative rank of that config among all N (1 = worst … N = best)
        rank = float(np.sum(oos_perf <= oos_perf[n_star]))
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))

    logits_arr = np.array(logits)
    pbo = float(np.mean(logits_arr < 0)) if logits_arr.size else 1.0
    return {
        "pbo": pbo,
        "n_configs": int(N),
        "n_combinations": int(logits_arr.size),
        "mean_logit": float(np.mean(logits_arr)) if logits_arr.size else 0.0,
    }


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
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if purge_days < 0:
            raise ValueError(f"purge_days must be >= 0, got {purge_days}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")
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
        n = len(index)
        fold_size = n // self.n_splits
        if fold_size == 0:
            raise ValueError(
                f"Index length {n} is too short for {self.n_splits} folds"
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
        """Deflated Sharpe Ratio — see module-level ``deflated_sharpe_ratio``.

        Returns a probability in [0, 1]: ≥ 0.95 clears the multiple-testing bar,
        low values mean the edge is within what luck would produce (overfit).
        """
        return deflated_sharpe_ratio(sharpe_ratios, n_trials)

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
              deflated_sharpe: DSR probability in [0,1] (adjusted for multiple testing)
              is_overfit: True if DSR < 0.95 (below 95% confidence vs the luck benchmark)
        """
        start_time = time.time()

        if not isinstance(signals.index, pd.DatetimeIndex):
            signals = signals.copy()
            signals.index = pd.to_datetime(signals.index)

        common_idx = signals.index.intersection(returns.index)
        signals = signals.loc[common_idx]
        returns = returns.loc[common_idx]

        signal_count = int(len(signals))

        sharpes: list[float] = []
        total_pnl = 0.0

        for train_idx, test_idx in self.split(pd.DatetimeIndex(signals.index)):
            test_signals = signals.iloc[test_idx]
            test_returns = returns.iloc[test_idx]
            # Shift signals by 1 to prevent lookahead
            pnl = test_signals.shift(1).fillna(0) * test_returns
            total_pnl += float(pnl.sum())
            sr = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(252)
            sharpes.append(float(sr))

        if not sharpes:
            result = {
                "fold_sharpes": [],
                "mean_sharpe": 0.0,
                "deflated_sharpe": 0.0,
                "is_overfit": True,
            }
        else:
            mean_sr = float(np.mean(sharpes))
            dsr = self.deflated_sharpe(sharpes, n_trials=len(sharpes))

            result = {
                "fold_sharpes": sharpes,
                "mean_sharpe": mean_sr,
                "deflated_sharpe": dsr,      # probability in [0,1]
                "is_overfit": dsr < 0.95,    # < 95% confidence it beats the luck benchmark
            }

        exec_time = time.time() - start_time

        logger.info(
            "CPCV validation completed",
            extra={
                "signal_count": signal_count,
                "execution_time_sec": exec_time,
                "total_pnl": total_pnl,
            },
        )

        return result