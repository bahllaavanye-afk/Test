"""Volatility-regime axis on the options/income desks.

Premium sellers (iron_condor, credit_spread_income, wheel, cash_secured_put,
vol_carry) face a HIGHER confidence bar in CALM vol — selling premium is thin
there (0DTE variance-risk-premium evidence) — but are never hard-blocked, so
the income desk still trades and leans into stressed regimes. Non-sellers are
unaffected. Pure functions, no network.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd  # noqa: E402

_MOD = Path(__file__).parent / "desk_order_placer.py"
_spec = importlib.util.spec_from_file_location("dop_vol_test", _MOD)
dop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dop)  # type: ignore[union-attr]


def _bars(vals):
    return pd.DataFrame({"close": vals})


def test_stressed_vs_calm_detection():
    # Calm: tiny steady drift. Stressed: alternating large moves in the last 20.
    calm = _bars([100 + i * 0.01 for i in range(120)])
    assert dop._detect_vol_regime_from_bars(calm) == "calm"

    px, vals = 100.0, []
    for i in range(120):
        px *= 1.05 if i % 2 == 0 else 0.955 if i >= 100 else 1.0005
        vals.append(px)
    assert dop._detect_vol_regime_from_bars(_bars(vals)) == "stressed"


def test_short_or_bad_series_is_calm():
    assert dop._detect_vol_regime_from_bars(_bars([100, 101])) == "calm"
    assert dop._detect_vol_regime_from_bars(None) == "calm"


def test_premium_seller_bar_raised_only_in_calm():
    seller = next(iter(dop._PREMIUM_SELLERS))
    base = 0.60
    # calm → raised by the bump
    assert dop._vol_adjusted_threshold(seller, base, "calm") == pytest.approx(base + dop._CALM_PREMIUM_THRESHOLD_BUMP)
    # stressed → unchanged
    assert dop._vol_adjusted_threshold(seller, base, "stressed") == base


def test_non_seller_never_adjusted():
    assert dop._vol_adjusted_threshold("momentum", 0.60, "calm") == 0.60
    assert dop._vol_adjusted_threshold("momentum", 0.60, "stressed") == 0.60


def test_income_structures_are_flagged_as_sellers():
    # The mleg income structures must all count as premium sellers.
    for name in ("iron_condor", "credit_spread_income"):
        assert name in dop._PREMIUM_SELLERS
