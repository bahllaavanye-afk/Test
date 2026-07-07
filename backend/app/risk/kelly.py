"""Kelly criterion position sizing with fractional Kelly for safety.

This module provides utility functions to calculate a fractional Kelly bet size and
to convert that bet size into an integer number of shares based on available equity
and the current price of the instrument. The implementation caps the position size
to limit exposure and reduce variance.
"""

from __future__ import annotations

import numpy as np


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Compute a fractional Kelly bet size.

    The full Kelly formula is::

        f = (p * b - q) / b

    where:
        p = win_rate (probability of a winning trade)
        q = 1 - p (probability of a losing trade)
        b = avg_win / avg_loss (payoff ratio)

    This function applies the ``fraction`` multiplier to the full Kelly result to
    reduce variance, and enforces a hard cap of 20 % of equity per position.

    Parameters
    ----------
    win_rate : float
        Probability of a winning trade (0 <= win_rate <= 1).
    avg_win : float
        Average profit per winning trade (as a decimal, e.g., 0.02 for 2%).
    avg_loss : float
        Average loss per losing trade (as a decimal, positive value).
    fraction : float, optional
        Fraction of the full Kelly bet to use (default is 0.25, i.e., 25 % Kelly).

    Returns
    -------
    float
        The fractional Kelly bet size, limited to a maximum of 0.20 (20 % of equity).
        Returns ``0.0`` if ``avg_loss`` is zero or ``win_rate`` is non‑positive.
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    f_full = (win_rate * b - q) / b
    f_full = max(0.0, f_full)
    return min(f_full * fraction, 0.20)


def size_from_kelly(
    equity: float,
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    price: float,
    max_pct: float = 0.05,
    kelly_fraction_pct: float = 0.25,
) -> int:
    """
    Determine an integer share count based on the Kelly criterion.

    The function first computes a fractional Kelly bet size using
    :func:`kelly_fraction`, then caps the resulting bet to ``max_pct`` of the total
    equity. The dollar amount is converted to a share quantity using the provided
    ``price``. At least one share is always returned.

    Parameters
    ----------
    equity : float
        Total account equity (in monetary units).
    win_rate : float
        Probability of a winning trade (0 <= win_rate <= 1).
    avg_win_pct : float
        Average winning trade return as a decimal (e.g., 0.02 for 2 %).
    avg_loss_pct : float
        Average losing trade return as a decimal (positive value).
    price : float
        Current price of the instrument.
    max_pct : float, optional
        Maximum allowed position size as a fraction of equity (default 0.05, i.e., 5 %).
    kelly_fraction_pct : float, optional
        Fraction of the full Kelly bet to apply (default 0.25, i.e., 25 % Kelly).

    Returns
    -------
    int
        The number of shares to trade, constrained to be at least one.
    """
    f = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_fraction_pct)
    f = min(f, max_pct)
    dollar_size = equity * f
    return max(1, int(dollar_size / price))