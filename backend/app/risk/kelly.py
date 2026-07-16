"""Kelly criterion position sizing utilities.

Provides functions to calculate fractional Kelly position sizes and convert them
to an integer number of shares based on available equity and price. The default
fractional Kelly factor reduces risk and caps the maximum allocation per trade.
"""

from __future__ import annotations

import numpy as np


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """Calculate a fractional Kelly position size.

    The full Kelly formula is:

        f = (p * b - q) / b

    where:
        p = win_rate (probability of a winning trade),
        b = avg_win / avg_loss (win/loss payout ratio),
        q = 1 - p (probability of a losing trade).

    This function applies a `fraction` (default 25 %) to the full Kelly value to
    reduce variance and caps the result at 20 % of equity per position.

    Args:
        win_rate: Probability of a winning trade (0 ≤ win_rate ≤ 1).
        avg_win: Average profit of winning trades (absolute value).
        avg_loss: Average loss of losing trades (absolute value, positive).
        fraction: Fraction of the full Kelly to use; typical values are 0.1–0.5.

    Returns:
        Fractional Kelly allocation as a decimal (e.g., 0.10 for 10 % of equity).
        Returns 0.0 if `avg_loss` is zero or `win_rate` is non‑positive.
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
    """Determine an integer share count sized by the Kelly criterion.

    The function first computes a fractional Kelly allocation using
    :func:`kelly_fraction`, then limits the allocation to `max_pct` of total
    equity. The resulting dollar amount is divided by the current price to
    obtain a share count, with a minimum of one share.

    Args:
        equity: Current portfolio equity (in currency units).
        win_rate: Probability of a winning trade.
        avg_win_pct: Average winning trade as a *percentage* of equity.
        avg_loss_pct: Average losing trade as a *percentage* of equity.
        price: Current price of the instrument.
        max_pct: Maximum proportion of equity to allocate to a single position.
            Defaults to 5 % (0.05).
        kelly_fraction_pct: Fraction of the full Kelly to apply.
            Defaults to 25 % (0.25).

    Returns:
        Number of shares to trade, rounded down to the nearest integer and
        constrained to be at least one.
    """
    f = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_fraction_pct)
    f = min(f, max_pct)
    dollar_size = equity * f
    return max(1, int(dollar_size / price))