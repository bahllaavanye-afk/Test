"""Utility functions for position sizing using the Kelly criterion.

This module provides helpers to compute a fractional Kelly factor based on
historical win rate, average win and loss percentages, and to convert that
factor into an integer number of shares given the current equity and price.
The implementation is deliberately lightweight and free of external
dependencies, suitable for use in low‑latency trading environments.
"""

import numpy as np
from typing import Final


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Compute a fractional Kelly position size.

    The full Kelly formula is::

        f = (p * b - q) / b

    where ``p`` is the win probability, ``b`` is the win‑to‑loss ratio
    (``avg_win / avg_loss``), and ``q = 1 - p``.  This function applies a
    safety ``fraction`` (default 25 %) to the full Kelly result and caps the
    final factor at 20 % of the account equity per position.

    Parameters
    ----------
    win_rate: float
        Historical win probability (0 ≤ win_rate ≤ 1).
    avg_win: float
        Average winning trade size expressed as a *fraction* of equity.
    avg_loss: float
        Average losing trade size expressed as a *fraction* of equity.
    fraction: float, optional
        Fraction of the full Kelly to use for safety.  Common values are
        0.25 (quarter Kelly) or 0.5 (half Kelly).  Defaults to 0.25.

    Returns
    -------
    float
        Fraction of equity to allocate to a single position, bounded between
        0.0 and 0.20.
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b: float = avg_win / avg_loss
    q: float = 1.0 - win_rate
    f_full: float = (win_rate * b - q) / b
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
    Calculate an integer share count based on the Kelly criterion.

    The function determines the Kelly fraction using :func:`kelly_fraction`,
    limits it to ``max_pct`` of the total equity, converts the resulting dollar
    allocation to a share count using the supplied ``price``, and ensures at
    least one share is returned.

    Parameters
    ----------
    equity: float
        Current portfolio equity (in currency units).
    win_rate: float
        Historical win probability of the strategy.
    avg_win_pct: float
        Average winning trade size as a fraction of equity.
    avg_loss_pct: float
        Average losing trade size as a fraction of equity.
    price: float
        Current price of the instrument.
    max_pct: float, optional
        Maximum allowed allocation of equity to a single position (default 5 %).
    kelly_fraction_pct: float, optional
        Safety fraction applied to the full Kelly result (default 0.25).

    Returns
    -------
    int
        Number of shares to trade, with a minimum of one.
    """
    f: float = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_fraction_pct)
    f = min(f, max_pct)
    dollar_size: float = equity * f
    return max(1, int(dollar_size / price))