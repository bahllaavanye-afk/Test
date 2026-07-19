"""
Value at Risk (VaR) and Conditional Value at Risk (CVaR/Expected Shortfall) utilities.

This module provides a simple implementation for calculating 1‑day VaR and CVaR
using either a historical (empirical) method or a parametric Gaussian
approach. The results are wrapped in a :class:`VaRResult` dataclass for easy
consumption by downstream risk reporting components.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """
    Container for VaR and CVaR calculation results.

    Attributes
    ----------
    var_95: float
        1‑day 95 % VaR expressed as a fraction of the portfolio (positive number).
    var_99: float
        1‑day 99 % VaR expressed as a fraction of the portfolio.
    cvar_95: float
        Expected shortfall at the 95 % confidence level.
    cvar_99: float
        Expected shortfall at the 99 % confidence level.
    method: str
        Calculation method used – ``'historical'`` or ``'parametric'`` (or a
        fallback identifier when data are insufficient).
    n_observations: int
        Number of return observations supplied to the calculation.
    portfolio_value: float
        Current portfolio value in USD.
    var_95_usd: float
        95 % VaR expressed in USD.
    var_99_usd: float
        99 % VaR expressed in USD.
    """

    var_95: float        # 1-day 95% VaR (as fraction of portfolio)
    var_99: float        # 1-day 99% VaR
    cvar_95: float       # Expected shortfall at 95% (CVaR)
    cvar_99: float       # Expected shortfall at 99%
    method: str          # 'historical' | 'parametric'
    n_observations: int
    portfolio_value: float
    var_95_usd: float    # VaR in USD
    var_99_usd: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a JSON‑serialisable dictionary.

        Returns
        -------
        dict
            Mapping with rounded percentage values, USD amounts and a short
            textual interpretation.
        """
        return {
            "var_95_pct": round(self.var_95 * 100, 3),
            "var_99_pct": round(self.var_99 * 100, 3),
            "cvar_95_pct": round(self.cvar_95 * 100, 3),
            "cvar_99_pct": round(self.cvar_99 * 100, 3),
            "var_95_usd": round(self.var_95_usd, 2),
            "var_99_usd": round(self.var_99_usd, 2),
            "method": self.method,
            "n_observations": self.n_observations,
            "interpretation": f"With 95% confidence, max 1-day loss ≤ ${self.var_95_usd:,.0f}",
        }


def historical_var(
    returns: List[float],
    portfolio_value: float,
    method: Literal["historical", "parametric"] = "historical",
) -> VaRResult:
    """
    Compute 1‑day VaR and CVaR from a series of daily returns.

    Parameters
    ----------
    returns : list[float]
        Daily return values where ``0.01`` represents a +1 % move.
    portfolio_value : float
        Current portfolio value in USD.
    method : {"historical", "parametric"}, optional
        The calculation approach:
        * ``"historical"`` – empirical quantiles from the supplied returns.
        * ``"parametric"`` – assumes returns follow a Gaussian distribution.

    Returns
    -------
    VaRResult
        Dataclass instance containing VaR/CVaR metrics, method metadata, and
        USD equivalents.
    """
    start_time = time.perf_counter()

    arr = np.array(returns, dtype=float)
    n = len(arr)

    if n < 10:
        # Not enough data — return conservative estimates
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

    if method == "historical":
        var_95 = float(-np.percentile(arr, 5))
        var_99 = float(-np.percentile(arr, 1))
        # CVaR = mean of losses beyond VaR threshold
        tail_95 = arr[arr < -var_95]
        tail_99 = arr[arr < -var_99]
        cvar_95 = float(-np.mean(tail_95)) if len(tail_95) > 0 else var_95 * 1.2
        cvar_99 = float(-np.mean(tail_99)) if len(tail_99) > 0 else var_99 * 1.2
    else:
        # Parametric (Gaussian)
        from scipy.stats import norm

        mu, sigma = float(np.mean(arr)), float(np.std(arr, ddof=1))
        var_95 = float(-(mu + norm.ppf(0.05) * sigma))
        var_99 = float(-(mu + norm.ppf(0.01) * sigma))
        # CVaR for Gaussian: E[X | X < q] = mu - sigma * φ(z) / Φ(z)
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