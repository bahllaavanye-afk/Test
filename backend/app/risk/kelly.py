"""Kelly criterion position sizing utilities.

This module provides functions to compute the Kelly fraction for a trading
strategy and to convert that fraction into an integer number of shares based
on available equity, price, and risk constraints. The implementation follows
the standard Kelly formula with an optional safety fraction to reduce
volatility, and enforces a hard cap on position size.
"""

import numpy as np
from typing import Final

__all__: Final = ["kelly_fraction", "size_from_kelly"]


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Compute the fractional Kelly position size.

    The classic Kelly formula is:

        f_full = (p * b - q) / b

    where:
        p = win_rate (probability of a winning trade)
        q = 1 - p (probability of a losing trade)
        b = avg_win / avg_loss (payoff ratio)

    This function returns a safety‑adjusted fraction of the full Kelly
    allocation. The `fraction` argument typically defaults to 0.25 (i.e.,
    25 % of the full Kelly) to limit drawdown risk. The final allocation is
    also capped at 20 % of equity per position.

    Args:
        win_rate: Probability of a winning trade (0 ≤ win_rate ≤ 1).
        avg_win: Average profit per winning trade (as a decimal, e.g., 0.02 for 2 %).
        avg_loss: Average loss per losing trade (as a positive decimal).
        fraction: Safety multiplier applied to the full Kelly result. Defaults to 0.25.

    Returns:
        A fractional Kelly value constrained between 0 and 0.20. Returns 0.0
        if `avg_loss` is zero or `win_rate` is non‑positive.
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    f_full = (win_rate * b - q) / b
    f_full = max(0.0, f_full)
    return min(f_full * fraction, 0.20)  # hard cap at 20% per position


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

    The function first computes the Kelly fraction using `kelly_fraction`,
    then applies an optional maximum equity percentage (`max_pct`). The
    resulting dollar allocation is divided by the current price to obtain
    the number of shares, with a minimum of one share enforced.

    Args:
        equity: Total account equity in monetary units.
        win_rate: Probability of a winning trade (0 ≤ win_rate ≤ 1).
        avg_win_pct: Average winning trade return as a decimal (e.g., 0.02 for 2 %).
        avg_loss_pct: Average losing trade return as a positive decimal.
        price: Current price of the instrument.
        max_pct: Upper bound on the portion of equity allocated to the position
                 (default 0.05, i.e., 5 %).
        kelly_fraction_pct: Safety multiplier passed to `kelly_fraction`
                            (default 0.25).

    Returns:
        The number of shares to trade, guaranteed to be at least one.
    """
    f = kelly_fraction(win_rate, avg_win_pct, avg_loss_pct, kelly_fraction_pct)
    f = min(f, max_pct)
    dollar_size = equity * f
    return max(1, int(dollar_size / price))