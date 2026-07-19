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


def _validate_series(name: str, series: Sequence) -> List[float]:
    """Validate that a series is numeric and convertible to float.

    Parameters
    ----------
    name : str
        Identifier used for logging.
    series : Sequence
        Input series to validate.

    Returns
    -------
    List[float]
        Cleaned list of floats.

    Raises
    ------
    TypeError
        If the series is not iterable or contains non‑numeric items.
    """
    if not isinstance(series, Sequence):
        raise TypeError(f"{name} must be a sequence, got {type(series)}")
    cleaned = []
    for idx, val in enumerate(series):
        try:
            cleaned.append(float(val))
        except (TypeError, ValueError) as e:
            raise TypeError(f"Non‑numeric value at index {idx} in {name}: {val}") from e
    return cleaned


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
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    try:
        portfolio = _validate_series("portfolio_returns", portfolio_returns)
        market = _validate_series("spy_returns", spy_returns)
        momentum = _validate_series("momentum_factor", momentum_factor) if momentum_factor else None
        low_vol = _validate_series("low_vol_factor", low_vol_factor) if low_vol_factor else None
    except TypeError as e:
        logger.error(
            "Invalid input series for factor exposure calculation",
            extra={"error": str(e)},
            exc_info=True,
        )
        return FactorExposure(
            market_beta=1.0,
            momentum_loading=0.0,
            low_vol_loading=0.0,
            size_loading=0.0,
            r_squared=0.0,
            alpha_annualized=0.0,
            tracking_error=0.02,
        )

    n = min(len(portfolio), len(market))
    if n < 20:
        logger.warning(
            "Insufficient data for factor exposure calculation",
            extra={"required_min": 20, "available_portfolio": len(portfolio), "available_market": len(market)},
        )
        return FactorExposure(
            market_beta=1.0,
            momentum_loading=0.0,
            low_vol_loading=0.0,
            size_loading=0.0,
            r_squared=0.0,
            alpha_annualized=0.0,
            tracking_error=0.02,
        )

    # ------------------------------------------------------------------
    # Build regression matrices
    # ------------------------------------------------------------------
    try:
        y = np.array(portfolio[-n:], dtype=float)
        X_cols = [np.ones(n, dtype=float), np.array(market[-n:], dtype=float)]
        col_names = ["alpha", "market"]

        if momentum and len(momentum) >= n:
            X_cols.append(np.array(momentum[-n:], dtype=float))
            col_names.append("momentum")
        else:
            logger.info(
                "Momentum factor not used (insufficient length or None)",
                extra={"required": n, "available": len(momentum) if momentum else 0},
            )
        if low_vol and len(low_vol) >= n:
            X_cols.append(np.array(low_vol[-n:], dtype=float))
            col_names.append("low_vol")
        else:
            logger.info(
                "Low‑vol factor not used (insufficient length or None)",
                extra={"required": n, "available": len(low_vol) if low_vol else 0},
            )

        X = np.column_stack(X_cols)
    except (ValueError, TypeError) as e:
        logger.error(
            "Failed to construct regression matrices",
            extra={"error": str(e), "n": n, "col_names": col_names},
            exc_info=True,
        )
        return FactorExposure(
            market_beta=1.0,
            momentum_loading=0.0,
            low_vol_loading=0.0,
            size_loading=0.0,
            r_squared=0.0,
            alpha_annualized=0.0,
            tracking_error=0.02,
        )

    # ------------------------------------------------------------------
    # Perform OLS regression
    # ------------------------------------------------------------------
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        logger.error(
            "Linear algebra error during OLS regression",
            extra={"error": str(e), "shape_X": X.shape, "shape_y": y.shape},
            exc_info=True,
        )
        return FactorExposure(
            market_beta=1.0,
            momentum_loading=0.0,
            low_vol_loading=0.0,
            size_loading=0.0,
            r_squared=0.0,
            alpha_annualized=0.0,
            tracking_error=0.02,
        )
    except Exception as e:
        logger.exception(
            "Unexpected error during factor exposure regression",
            extra={"error": str(e), "shape_X": X.shape, "shape_y": y.shape},
        )
        return FactorExposure(
            market_beta=1.0,
            momentum_loading=0.0,
            low_vol_loading=0.0,
            size_loading=0.0,
            r_squared=0.0,
            alpha_annualized=0.0,
            tracking_error=0.02,
        )

    # ------------------------------------------------------------------
    # Extract coefficients and compute diagnostics
    # ------------------------------------------------------------------
    alpha_daily = float(coeffs[0])
    market_beta = float(coeffs[1])
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