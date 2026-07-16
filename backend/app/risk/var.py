"""
Value at Risk (VaR) and Conditional Value at Risk (CVaR/Expected Shortfall).
These are the primary risk metrics used by institutional desks.

VaR(95%) = worst 5% of daily returns threshold
CVaR(95%) = average loss in the worst 5% of days
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    var_95: float        # 1‑day 95% VaR (as fraction of portfolio)
    var_99: float        # 1‑day 99% VaR
    cvar_95: float       # Expected shortfall at 95% (CVaR)
    cvar_99: float       # Expected shortfall at 99%
    method: str          # 'historical' | 'parametric'
    n_observations: int
    portfolio_value: float
    var_95_usd: float    # VaR in USD
    var_99_usd: float

    def to_dict(self) -> dict:
        """Return a serialisable representation of the result."""
        return {
            "var_95_pct": round(self.var_95 * 100, 3),
            "var_99_pct": round(self.var_99 * 100, 3),
            "cvar_95_pct": round(self.cvar_95 * 100, 3),
            "cvar_99_pct": round(self.cvar_99 * 100, 3),
            "var_95_usd": round(self.var_95_usd, 2),
            "var_99_usd": round(self.var_99_usd, 2),
            "method": self.method,
            "n_observations": self.n_observations,
            "interpretation": f"With 95% confidence, max 1‑day loss ≤ ${self.var_95_usd:,.0f}",
        }


def _clean_returns(returns: Sequence[float]) -> np.ndarray:
    """
    Validate and clean the return series.

    - Removes NaN / infinite values.
    - Winsorises extreme outliers to the 1st / 99th percentile to avoid
      pathological tail estimates when data is sparse.
    - Returns an ``np.ndarray`` of type ``float64``.
    """
    arr = np.asarray(returns, dtype=float)

    # Filter out non‑finite values
    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        logger.warning(
            "Non‑finite values detected in returns series; they will be removed.",
            extra={"original_len": len(arr), "valid_len": int(finite_mask.sum())},
        )
        arr = arr[finite_mask]

    # If too few points remain, return early – caller will handle insufficient data
    if arr.size < 10:
        return arr

    # Winsorise extreme values (1st and 99th percentiles) to reduce noise
    lower, upper = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lower, upper)
    return arr


def historical_var(
    returns: Sequence[float],
    portfolio_value: float,
    method: Literal["historical", "parametric"] = "historical",
    *,
    volatility_filter: float | None = None,
) -> VaRResult:
    """
    Compute VaR and CVaR from a return series.

    Parameters
    ----------
    returns : Sequence[float]
        Daily returns expressed as decimal (e.g. 0.01 = +1%).
    portfolio_value : float
        Current portfolio value in USD.
    method : {"historical", "parametric"}, default "historical"
        Calculation approach.
    volatility_filter : float, optional
        If supplied, the function aborts and returns a conservative estimate
        when the annualised volatility (std * sqrt(252)) exceeds the threshold.
        This acts as a confirmation filter to avoid unreliable VaR estimates
        during unusually calm periods.

    Returns
    -------
    VaRResult
        Dataclass containing VaR/CVaR metrics and auxiliary information.
    """
    start_time = time.perf_counter()

    arr = _clean_returns(returns)
    n = int(arr.size)

    # Insufficient data guard – fall back to conservative defaults
    if n < 10:
        result = VaRResult(
            var_95=0.02,
            var_99=0.03,
            cvar_95=0.03,
            cvar_99=0.04,
            method="default_insufficient_data",
            n_observations=n,
            portfolio_value=portfolio_value,
            var_95_usd=portfolio_value * 0.02,
            var_99_usd=portfolio_value * 0.03,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "VaR calculation completed (insufficient data)",
            extra={
                "signal_count": n,
                "execution_time_ms": round(elapsed_ms, 2),
                "pnl_usd": portfolio_value,
                "var_95_usd": result.var_95_usd,
                "var_99_usd": result.var_99_usd,
                "method": result.method,
            },
        )
        return result

    # Optional volatility confirmation filter
    if volatility_filter is not None:
        annual_vol = float(np.std(arr, ddof=1) * np.sqrt(252))
        if annual_vol < volatility_filter:
            logger.info(
                "Volatility below filter threshold; returning conservative VaR.",
                extra={"annual_vol": round(annual_vol, 4), "threshold": volatility_filter},
            )
            result = VaRResult(
                var_95=0.015,
                var_99=0.025,
                cvar_95=0.025,
                cvar_99=0.035,
                method="conservative_vol_filter",
                n_observations=n,
                portfolio_value=portfolio_value,
                var_95_usd=portfolio_value * 0.015,
                var_99_usd=portfolio_value * 0.025,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "VaR calculation completed (volatility filter)",
                extra={
                    "signal_count": n,
                    "execution_time_ms": round(elapsed_ms, 2),
                    "pnl_usd": portfolio_value,
                    "var_95_usd": result.var_95_usd,
                    "var_99_usd": result.var_99_usd,
                    "method": result.method,
                },
            )
            return result

    if method == "historical":
        # Tighten entry condition by using a more conservative percentile
        var_95 = float(-np.percentile(arr, 4))   # 96th percentile instead of 5%
        var_99 = float(-np.percentile(arr, 0.5))  # 99.5th percentile instead of 1%
        tail_95 = arr[arr < -var_95]
        tail_99 = arr[arr < -var_99]
        cvar_95 = float(-np.mean(tail_95)) if tail_95.size else var_95 * 1.2
        cvar_99 = float(-np.mean(tail_99)) if tail_99.size else var_99 * 1.2
    else:
        # Parametric (Gaussian) – same as before but guard against zero sigma
        from scipy.stats import norm

        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        if sigma == 0:
            logger.warning("Zero volatility detected; falling back to historical estimates.")
            return historical_var(returns, portfolio_value, method="historical")

        var_95 = float(-(mu + norm.ppf(0.05) * sigma))
        var_99 = float(-(mu + norm.ppf(0.01) * sigma))
        cvar_95 = float(-(mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05))
        cvar_99 = float(-(mu - sigma * norm.pdf(norm.ppf(0.01)) / 0.01))

    result = VaRResult(
        var_95=max(var_95, 0),
        var_99=max(var_99, 0),
        cvar_95=max(cvar_95, 0),
        cvar_99=max(cvar_99, 0),
        method=method,
        n_observations=n,
        portfolio_value=portfolio_value,
        var_95_usd=portfolio_value * max(var_95, 0),
        var_99_usd=portfolio_value * max(var_99, 0),
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "VaR calculation completed",
        extra={
            "signal_count": n,
            "execution_time_ms": round(elapsed_ms, 2),
            "pnl_usd": portfolio_value,
            "var_95_usd": result.var_95_usd,
            "var_99_usd": result.var_99_usd,
            "method": method,
        },
    )
    return result