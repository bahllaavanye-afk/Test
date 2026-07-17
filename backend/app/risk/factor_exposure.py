"""
Factor exposure analysis — measures how much of portfolio returns
are explained by common risk factors (market beta, momentum, low‑vol).

Standard at all hedge funds. Uniquely missing from open‑source bots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FactorExposure:
    """Container for the results of a factor‑exposure regression.

    Attributes
    ----------
    market_beta : float
        Sensitivity to the market (SPY). 1.0 corresponds to full market exposure,
        0.0 to a market‑neutral portfolio.
    momentum_loading : float
        Loading on the 12‑1 month momentum factor.
    low_vol_loading : float
        Loading on the low‑volatility factor.
    size_loading : float
        Loading on a size factor (SMB‑like). Not currently computed.
    r_squared : float
        Proportion of variance explained by the regression model.
    alpha_annualized : float
        Annualized Jensen's alpha (excess return versus the factor model).
    tracking_error : float
        Standard deviation of the residuals (daily) from the factor model.
    """

    market_beta: float
    momentum_loading: float
    low_vol_loading: float
    size_loading: float
    r_squared: float
    alpha_annualized: float
    tracking_error: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the dataclass to a dictionary with rounded values.

        Returns
        -------
        dict
            Mapping of field names to human‑readable numbers and an interpretation
            string.
        """
        return {
            "market_beta": round(self.market_beta, 3),
            "momentum_loading": round(self.momentum_loading, 3),
            "low_vol_loading": round(self.low_vol_loading, 3),
            "size_loading": round(self.size_loading, 3),
            "r_squared": round(self.r_squared, 3),
            "alpha_annualized_pct": round(self.alpha_annualized * 100, 2),
            "tracking_error_pct": round(self.tracking_error * 100, 2),
            "interpretation": _interpret(self),
        }


def _interpret(fe: FactorExposure) -> str:
    """Generate a short textual interpretation of a factor‑exposure result.

    Parameters
    ----------
    fe : FactorExposure
        The factor exposure object to interpret.

    Returns
    -------
    str
        Human‑readable description summarising the dominant exposures.
    """
    parts = []
    if abs(fe.market_beta) < 0.2:
        parts.append("Market-neutral")
    elif fe.market_beta > 0.8:
        parts.append(f"High market exposure (β={fe.market_beta:.2f})")
    if fe.momentum_loading > 0.3:
        parts.append("Momentum tilt")
    elif fe.momentum_loading < -0.3:
        parts.append("Contrarian/mean-reversion tilt")
    if fe.alpha_annualized > 0.05:
        parts.append(f"Positive alpha ({fe.alpha_annualized*100:.1f}% ann)")
    return ", ".join(parts) if parts else "Balanced factor exposure"


def _default_exposure() -> FactorExposure:
    """Create a safe default FactorExposure used when inputs are invalid."""
    return FactorExposure(
        market_beta=1.0,
        momentum_loading=0.0,
        low_vol_loading=0.0,
        size_loading=0.0,
        r_squared=0.0,
        alpha_annualized=0.0,
        tracking_error=0.02,
    )


def _validate_series(series: Optional[Sequence[float]], name: str) -> Optional[np.ndarray]:
    """Validate that a series is non‑empty and convertible to a NumPy array.

    Returns ``None`` if the series is ``None``, empty, or cannot be converted.
    """
    if series is None:
        return None
    if not isinstance(series, (list, tuple, np.ndarray)):
        logger.warning(
            "Invalid type for %s; expected list/tuple/ndarray",
            name,
            extra={"type": type(series)},
        )
        return None
    if len(series) == 0:
        logger.warning("%s is empty; it will be ignored", name)
        return None
    try:
        arr = np.array(series, dtype=float)
        if np.isnan(arr).any():
            logger.warning("%s contains NaN values; they will be treated as missing", name)
            return None
        return arr
    except Exception as e:  # pragma: no cover
        logger.error(
            "Failed to convert %s to NumPy array",
            name,
            extra={"error": str(e)},
        )
        return None


def compute_factor_exposure(
    portfolio_returns: List[float],
    spy_returns: List[float],
    momentum_factor: Optional[List[float]] = None,
    low_vol_factor: Optional[List[float]] = None,
) -> FactorExposure:
    """Estimate factor exposures using ordinary least‑squares regression.

    The regression model is:

        portfolio = α + β_market * SPY + β_momentum * momentum + β_low_vol * low_vol + ε

    Only the market factor is mandatory; the momentum and low‑vol factors are
    included when sufficient data are supplied.

    Parameters
    ----------
    portfolio_returns : list[float]
        Daily portfolio returns.
    spy_returns : list[float]
        Daily SPY returns representing the market factor.
    momentum_factor : list[float] | None, optional
        Daily returns of a momentum factor. Ignored if ``None`` or too short.
    low_vol_factor : list[float] | None, optional
        Daily returns of a low‑volatility factor. Ignored if ``None`` or too short.

    Returns
    -------
    FactorExposure
        The regression coefficients and diagnostics wrapped in a ``FactorExposure`` instance.
    """
    # Basic validation of mandatory inputs
    if not portfolio_returns or not spy_returns:
        logger.warning(
            "Missing mandatory return series; returning default exposure",
            extra={"portfolio_len": len(portfolio_returns) if portfolio_returns else 0,
                   "spy_len": len(spy_returns) if spy_returns else 0},
        )
        return _default_exposure()

    # Convert mandatory series to NumPy arrays (validation ensures non‑empty)
    y_arr = _validate_series(portfolio_returns, "portfolio_returns")
    x_market_arr = _validate_series(spy_returns, "spy_returns")
    if y_arr is None or x_market_arr is None:
        return _default_exposure()

    # Determine usable sample size
    n = min(len(y_arr), len(x_market_arr))
    if n < 20:
        logger.warning(
            "Insufficient data for factor exposure calculation",
            extra={"required_min": 20, "available": n},
        )
        return _default_exposure()

    # Prepare regression matrices
    col_names = ["alpha", "market"]
    try:
        y = y_arr[-n:]
        X_cols = [np.ones(n, dtype=float), x_market_arr[-n:]]
        # Optional momentum factor
        mom_arr = _validate_series(momentum_factor, "momentum_factor")
        if mom_arr is not None and len(mom_arr) >= n:
            X_cols.append(mom_arr[-n:])
            col_names.append("momentum")
        # Optional low‑vol factor
        low_vol_arr = _validate_series(low_vol_factor, "low_vol_factor")
        if low_vol_arr is not None and len(low_vol_arr) >= n:
            X_cols.append(low_vol_arr[-n:])
            col_names.append("low_vol")

        X = np.column_stack(X_cols)
    except Exception as e:  # pragma: no cover
        logger.error(
            "Failed to construct regression matrices",
            extra={"error": str(e), "n": n, "col_names": col_names},
        )
        return _default_exposure()

    # Perform OLS regression
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        logger.error(
            "Linear algebra error during OLS regression",
            extra={"error": str(e), "shape_X": X.shape, "shape_y": y.shape},
        )
        return _default_exposure()
    except Exception as e:  # pragma: no cover
        logger.exception(
            "Unexpected error during factor exposure regression",
            extra={"error": str(e), "shape_X": X.shape, "shape_y": y.shape},
        )
        return _default_exposure()

    # Extract coefficients safely
    alpha_daily = float(coeffs[0]) if len(coeffs) > 0 else 0.0
    market_beta = float(coeffs[1]) if len(coeffs) > 1 else 0.0
    momentum_loading = float(coeffs[2]) if len(coeffs) > 2 else 0.0
    low_vol_loading = float(coeffs[3]) if len(coeffs) > 3 else 0.0

    # Goodness of fit
    y_hat = X @ coeffs
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    tracking_error = float(np.std(y - y_hat, ddof=1))

    return FactorExposure(
        market_beta=market_beta,
        momentum_loading=momentum_loading,
        low_vol_loading=low_vol_loading,
        size_loading=0.0,   # would need SMB factor data
        r_squared=max(0.0, r_squared),
        alpha_annualized=alpha_daily * 252,
        tracking_error=tracking_error,
    )