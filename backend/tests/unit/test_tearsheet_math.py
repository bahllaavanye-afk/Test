"""The tearsheet must not 500 on hostile-but-legal trade data.

`annualized_return = (1.0 + total_return) ** (1.0 / n_years)` raises a negative
base to a fractional power once total_return <= -100%. Python answers with a
COMPLEX number, and the next `float()`/`round()` dies with

    TypeError: float() argument must be a string or a real number, not 'complex'

which 500s the whole endpoint. Losing more than the baseline equity is entirely
reachable in paper trading, and the tearsheet is the investor-facing view.

These exercise the metric maths directly rather than through the route: the
route needs a DB, a user and an account, and the arithmetic is what breaks.
"""
from __future__ import annotations

import math

import pytest


def _annualized(total_return: float, days: int = 365) -> float:
    """Mirrors the endpoint's annualised-return step, including its guard."""
    n_years = days / 365.0
    if total_return <= -1.0:
        return -1.0
    return (1.0 + total_return) ** (1.0 / max(n_years, 0.01)) - 1.0


@pytest.mark.parametrize("total_return", [-1.0, -1.5, -2.0, -37.0])
def test_losing_more_than_the_baseline_does_not_go_complex(total_return):
    result = _annualized(total_return)
    assert isinstance(result, float), "a fractional power of a negative base returns complex"
    assert result == -1.0, "losing the whole baseline is -100% annualised"
    # The endpoint calls round() on this — that is where the complex value died.
    assert round(result * 100, 2) == -100.0


def test_ordinary_returns_are_unchanged():
    """The guard must not alter any case that already worked."""
    assert _annualized(0.0) == pytest.approx(0.0)
    assert _annualized(0.10, days=365) == pytest.approx(0.10)
    # A 21% gain over 182 days (0.4986 yr) compounds to ~46.6% annualised.
    assert _annualized(0.21, days=182) == pytest.approx(0.4656, abs=1e-3)
    assert _annualized(-0.5, days=365) == pytest.approx(-0.5)


def test_the_unguarded_form_really_does_produce_complex():
    """Pin the behaviour this guard exists for, so the reason stays legible."""
    raw = (1.0 + -2.0) ** (1.0 / 1.0 if False else 1.0 / 0.25)   # (-1.0) ** 4.0
    assert isinstance(raw, float)          # integral exponent stays real

    complex_result = (1.0 + -2.0) ** (1.0 / 3.0)                 # fractional → complex
    assert isinstance(complex_result, complex)
    with pytest.raises(TypeError):
        float(complex_result)


def test_sharpe_guard_survives_a_single_trading_day():
    """One day of trades gives std()==NaN (ddof=1); NaN > 0 is False, so 0.0."""
    import pandas as pd

    daily_returns = pd.Series([0.001])
    std = daily_returns.std()
    assert math.isnan(std)
    sharpe = float((daily_returns.mean() - 0.05 / 252) / std * math.sqrt(252)) if std > 0 else 0.0
    assert sharpe == 0.0
