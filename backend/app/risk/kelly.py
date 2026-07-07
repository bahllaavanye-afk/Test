"""Kelly criterion position sizing with fractional Kelly for safety.

Enhanced with tighter entry validation and exit helpers to improve signal quality.
"""
import numpy as np
from typing import Optional


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator/denominator safely, falling back to default on zero division."""
    return numerator / denominator if denominator != 0 else default


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
    *,
    min_win_rate: float = 0.55,
    min_reward_ratio: float = 1.5,
    max_volatility: Optional[float] = None,
    volatility: Optional[float] = None,
) -> float:
    """
    Compute a fractional Kelly position size.

    The function applies additional entry filters:
      * win_rate must exceed ``min_win_rate`` (default 55%).
      * reward ratio (avg_win / avg_loss) must exceed ``min_reward_ratio``.
      * if ``volatility`` is supplied, it must be below ``max_volatility``.

    Parameters
    ----------
    win_rate: float
        Historical win probability (0‑1).
    avg_win: float
        Average winning return (as a decimal, e.g., 0.02 for 2%).
    avg_loss: float
        Average losing return (as a decimal, positive value).
    fraction: float, default 0.25
        Fraction of the full Kelly to apply.
    min_win_rate: float, default 0.55
        Minimum win rate required to consider the signal.
    min_reward_ratio: float, default 1.5
        Minimum reward‑to‑risk ratio required.
    max_volatility: float | None, optional
        Upper bound on volatility; if ``volatility`` is None the check is skipped.
    volatility: float | None, optional
        Current volatility measure (e.g., standard deviation of returns).

    Returns
    -------
    float
        Fraction of equity to risk (capped at 20% per position).
        Returns 0.0 if any entry filter fails.
    """
    # Basic sanity checks
    if not (0.0 < win_rate <= 1.0):
        return 0.0
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    # Entry filters
    reward_ratio = _safe_divide(avg_win, avg_loss, default=0.0)
    if win_rate < min_win_rate or reward_ratio < min_reward_ratio:
        return 0.0
    if volatility is not None and max_volatility is not None:
        if volatility > max_volatility:
            return 0.0

    # Full Kelly calculation
    b = reward_ratio
    q = 1.0 - win_rate
    f_full = (win_rate * b - q) / b
    f_full = max(0.0, f_full)

    # Apply fractional Kelly and hard cap
    f = f_full * fraction
    return min(f, 0.20)


def size_from_kelly(
    equity: float,
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    price: float,
    max_pct: float = 0.05,
    kelly_fraction_pct: float = 0.25,
    *,
    volatility: Optional[float] = None,
) -> int:
    """
    Convert Kelly fraction to an integer share count.

    The function respects the same entry filters as ``kelly_fraction``.
    If the filters reject the signal, a size of 0 is returned (no position).

    Parameters
    ----------
    equity: float
        Total account equity in currency units.
    win_rate: float
        Historical win probability (0‑1).
    avg_win_pct: float
        Average winning return as a decimal (e.g., 0.02 for 2%).
    avg_loss_pct: float
        Average losing return as a decimal (positive value).
    price: float
        Current price of the instrument.
    max_pct: float, default 0.05
        Upper bound on Kelly fraction before applying the fractional factor.
    kelly_fraction_pct: float, default 0.25
        Fraction of the full Kelly to use.
    volatility: float | None, optional
        Current volatility measure used for entry validation.

    Returns
    -------
    int
        Number of shares to trade (minimum 1 if signal passes, otherwise 0).
    """
    if equity <= 0 or price <= 0:
        return 0

    f = kelly_fraction(
        win_rate,
        avg_win_pct,
        avg_loss_pct,
        fraction=kelly_fraction_pct,
        volatility=volatility,
    )
    if f == 0.0:
        return 0

    f = min(f, max_pct)
    dollar_size = equity * f
    share_count = int(dollar_size / price)

    return max(1, share_count)


def is_entry_valid(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    *,
    volatility: Optional[float] = None,
    min_win_rate: float = 0.55,
    min_reward_ratio: float = 1.5,
    max_volatility: Optional[float] = None,
) -> bool:
    """
    Stand‑alone entry filter that mirrors the checks performed inside ``kelly_fraction``.

    Returns ``True`` when the signal satisfies the configured thresholds.
    """
    if not (0.0 < win_rate <= 1.0):
        return False
    if avg_win <= 0 or avg_loss <= 0:
        return False

    reward_ratio = _safe_divide(avg_win, avg_loss, default=0.0)
    if win_rate < min_win_rate or reward_ratio < min_reward_ratio:
        return False
    if volatility is not None and max_volatility is not None and volatility > max_volatility:
        return False
    return True


def should_exit(
    current_profit_pct: float,
    position_kelly_frac: float,
    *,
    profit_target_multiplier: float = 2.0,
    loss_limit_multiplier: float = 0.5,
) -> bool:
    """
    Determine whether a position should be exited based on profit or loss thresholds
    relative to the Kelly fraction used for sizing.

    Parameters
    ----------
    current_profit_pct: float
        Realised/unrealised profit as a decimal (e.g., 0.03 for 3%).
    position_kelly_frac: float
        Kelly fraction that was applied to the position (0‑1).
    profit_target_multiplier: float, default 2.0
        Exit when profit exceeds ``profit_target_multiplier`` × Kelly fraction.
    loss_limit_multiplier: float, default 0.5
        Exit when loss exceeds ``loss_limit_multiplier`` × Kelly fraction (negative side).

    Returns
    -------
    bool
        ``True`` if exit conditions are met.
    """
    if position_kelly_frac <= 0:
        return False

    profit_target = position_kelly_frac * profit_target_multiplier
    loss_limit = -position_kelly_frac * loss_limit_multiplier

    return current_profit_pct >= profit_target or current_profit_pct <= loss_limit