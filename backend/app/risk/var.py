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
from typing import Literal, Iterable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    var_95: float        # 1-day 95% VaR (as fraction of portfolio)
    var_99: float        # 1-day 99% VaR
    cvar_95: float       # Expected shortfall at 95% (CVaR)
    cvar_99: float       # Expected shortfall at 99%
    method: str          # 'historical' | 'parametric'
    n_observations: int
    portfolio_value: float
    var_95_usd: float    # VaR in USD
    var_99_usd: float

    def to_dict(self) -> dict:
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
    returns: Iterable[float],
    portfolio_value: float,
    method: Literal["historical", "parametric"] = "historical",
) -> VaRResult:
    """
    Compute VaR and CVaR from a return series.

    Args:
        returns: Iterable of daily returns (e.g. 0.01 = +1%)
        portfolio_value: Current portfolio value in USD
        method: 'historical' (empirical) or 'parametric' (Gaussian)
    """
    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(portfolio_value, (int, float)):
        logger.error(
            "Invalid portfolio_value type",
            extra={"portfolio_value": portfolio_value, "type": type(portfolio_value)},
        )
        raise TypeError("portfolio_value must be a numeric type")

    if portfolio_value <= 0:
        logger.error(
            "Non‑positive portfolio_value",
            extra={"portfolio_value": portfolio_value},
        )
        raise ValueError("portfolio_value must be greater than zero")

    try:
        arr = np.array(list(returns), dtype=float)
    except (TypeError, ValueError) as exc:
        logger.exception(
            "Failed to convert returns to numpy array",
            extra={"error": str(exc)},
        )
        raise ValueError("returns must be an iterable of numeric values") from exc

    if arr.size == 0:
        logger.error(
            "Empty returns sequence",
            extra={"returns_length": arr.size},
        )
        raise ValueError("returns sequence cannot be empty")

    if np.isnan(arr).any():
        logger.error(
            "NaN values detected in returns",
            extra={"nan_count": int(np.isnan(arr).sum())},
        )
        raise ValueError("returns contain NaN values")

    n = len(arr)

    # ------------------------------------------------------------------
    # Insufficient data handling (conservative defaults)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Core calculation with error handling
    # ------------------------------------------------------------------
    try:
        if method == "historical":
            var_95 = float(-np.percentile(arr, 5))
            var_99 = float(-np.percentile(arr, 1))
            tail_95 = arr[arr < -var_95]
            tail_99 = arr[arr < -var_99]
            cvar_95 = float(-np.mean(tail_95)) if len(tail_95) > 0 else var_95 * 1.2
            cvar_99 = float(-np.mean(tail_99)) if len(tail_99) > 0 else var_99 * 1.2
        else:
            # Parametric (Gaussian)
            try:
                from scipy.stats import norm
            except ImportError as exc:
                logger.exception(
                    "SciPy is required for parametric VaR calculation",
                    extra={"error": str(exc)},
                )
                raise RuntimeError("SciPy is not installed") from exc

            mu, sigma = float(np.mean(arr)), float(np.std(arr, ddof=1))
            var_95 = float(-(mu + norm.ppf(0.05) * sigma))
            var_99 = float(-(mu + norm.ppf(0.01) * sigma))
            cvar_95 = float(-(mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05))
            cvar_99 = float(-(mu - sigma * norm.pdf(norm.ppf(0.01)) / 0.01))
    except Exception as exc:
        logger.exception(
            "Error during VaR computation",
            extra={
                "method": method,
                "signal_count": n,
                "portfolio_value": portfolio_value,
                "error": str(exc),
            },
        )
        raise RuntimeError("Failed to compute VaR") from exc

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