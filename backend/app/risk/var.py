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
from typing import Literal, Sequence, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Container for VaR and CVaR calculation results.

    Attributes
    ----------
    var_95 : float
        1‑day 95 % VaR expressed as a fraction of the portfolio (positive value).
    var_99 : float
        1‑day 99 % VaR expressed as a fraction of the portfolio.
    cvar_95 : float
        Expected shortfall at the 95 % confidence level.
    cvar_99 : float
        Expected shortfall at the 99 % confidence level.
    method : str
        Calculation method used – ``'historical'`` or ``'parametric'``.
    n_observations : int
        Number of return observations used in the calculation.
    portfolio_value : float
        Portfolio value (USD) supplied to the calculator.
    var_95_usd : float
        95 % VaR expressed in USD.
    var_99_usd : float
        99 % VaR expressed in USD.
    """

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    method: str
    n_observations: int
    portfolio_value: float
    var_95_usd: float
    var_99_usd: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON‑serialisable representation of the result.

        The percentages are rounded to three decimal places, USD values to two
        decimals, and a short interpretation string is added.

        Returns
        -------
        dict
            Mapping of metric names to their formatted values.
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
    returns: Sequence[float],
    portfolio_value: float,
    method: Literal["historical", "parametric"] = "historical",
) -> VaRResult:
    """Compute VaR and CVaR from a series of portfolio returns.

    Parameters
    ----------
    returns : Sequence[float]
        Daily portfolio returns expressed as decimals (e.g. ``0.01`` for +1 %).
    portfolio_value : float
        Current portfolio value in USD.
    method : {"historical", "parametric"}, default ``"historical"``
        Calculation approach – ``"historical"`` uses the empirical distribution,
        while ``"parametric"`` assumes a Gaussian distribution.

    Returns
    -------
    VaRResult
        Populated result object containing VaR, CVaR and auxiliary metadata.
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
        # CVaR for Gaussian: E[X | X < q] = mu - sigma * phi(z) / Phi(z)
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