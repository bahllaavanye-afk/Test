"""
Factor exposure analysis — measures how much of portfolio returns
are explained by common risk factors (market beta, momentum, low-vol).

Standard at all hedge funds. Uniquely missing from open-source bots.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FactorExposure:
    market_beta: float          # sensitivity to SPY (1.0 = market, 0.0 = market-neutral)
    momentum_loading: float     # factor loading on 12-1 month momentum
    low_vol_loading: float      # loading on low-volatility factor
    size_loading: float         # small vs large cap bias (SMB-like)
    r_squared: float            # variance explained by factors
    alpha_annualized: float     # Jensen's alpha (excess return vs model)
    tracking_error: float       # std of residuals vs factor model

    def to_dict(self) -> dict:
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


def compute_factor_exposure(
    portfolio_returns: list[float],
    spy_returns: list[float],
    momentum_factor: Optional[list[float]] = None,
    low_vol_factor: Optional[list[float]] = None,
) -> FactorExposure:
    """
    OLS regression of portfolio returns on factor returns.

    Args:
        portfolio_returns: Daily portfolio returns
        spy_returns: SPY daily returns (market factor)
        momentum_factor: Optional momentum factor returns
        low_vol_factor: Optional low-volatility factor returns
    """
    n = min(len(portfolio_returns), len(spy_returns))
    if n < 20:
        return FactorExposure(
            market_beta=1.0, momentum_loading=0.0, low_vol_loading=0.0,
            size_loading=0.0, r_squared=0.0, alpha_annualized=0.0, tracking_error=0.02
        )

    y = np.array(portfolio_returns[-n:])
    X_cols = [np.ones(n), np.array(spy_returns[-n:])]
    col_names = ["alpha", "market"]

    if momentum_factor and len(momentum_factor) >= n:
        X_cols.append(np.array(momentum_factor[-n:]))
        col_names.append("momentum")
    if low_vol_factor and len(low_vol_factor) >= n:
        X_cols.append(np.array(low_vol_factor[-n:]))
        col_names.append("low_vol")

    X = np.column_stack(X_cols)
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        return FactorExposure(market_beta=1.0, momentum_loading=0.0, low_vol_loading=0.0,
                              size_loading=0.0, r_squared=0.0, alpha_annualized=0.0, tracking_error=0.02)

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
        r_squared=max(0, r_squared),
        alpha_annualized=alpha_daily * 252,
        tracking_error=tracking_error,
    )
