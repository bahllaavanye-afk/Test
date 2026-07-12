"""Vol-targeted sizing (Moreira-Muir): size scales inversely with realized vol."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pandas")
import numpy as np
import pandas as pd

_MOD = Path(__file__).parent / "desk_order_placer.py"


def _load():
    spec = importlib.util.spec_from_file_location("dop_vol_test", _MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


dop = _load()


def _bars(daily_vol: float, n: int = 60) -> pd.DataFrame:
    rs = np.random.RandomState(42)
    closes = 100 * np.exp(np.cumsum(rs.normal(0, daily_vol, n)))
    return pd.DataFrame({"close": closes})


def test_calm_market_sizes_up_capped_at_2x():
    calm = _bars(0.002)   # ~3% annualized << 20% target
    assert dop._vol_scalar(calm) == 2.0


def test_violent_market_sizes_down_floored_at_half():
    wild = _bars(0.05)    # ~80% annualized >> 20% target
    assert dop._vol_scalar(wild) == 0.5


def test_at_target_vol_scalar_near_one():
    at_target = _bars(0.0126)  # ≈ 20% annualized
    assert 0.7 <= dop._vol_scalar(at_target) <= 1.4


def test_short_or_missing_bars_fall_back_to_one():
    assert dop._vol_scalar(_bars(0.01, n=10)) == 1.0     # too short
    assert dop._vol_scalar(None) == 1.0                   # absent
    flat = pd.DataFrame({"close": [100.0] * 60})          # zero vol
    assert dop._vol_scalar(flat) == 1.0


def test_kelly_notional_scales_with_vol():
    base = dop._kelly_notional(100_000, 0.90)                       # no bars
    calm = dop._kelly_notional(100_000, 0.90, bars=_bars(0.002))
    wild = dop._kelly_notional(100_000, 0.90, bars=_bars(0.05))
    assert calm == pytest.approx(base * 2.0)
    assert wild == pytest.approx(base * 0.5)
    assert wild >= 50.0                                             # floor holds


def test_kelly_without_bars_unchanged_behavior():
    # regression: legacy call signature still works and is deterministic
    assert dop._kelly_notional(100_000, 0.90) == dop._kelly_notional(100_000, 0.90)


# ── Daily loss circuit breaker ────────────────────────────────────────────────

def test_loss_cap_triggers_beyond_2pct():
    assert dop.daily_loss_cap_hit(97_900.0, 100_000.0, cap=0.02) is True    # -2.1%
    assert dop.daily_loss_cap_hit(98_100.0, 100_000.0, cap=0.02) is False   # -1.9%
    assert dop.daily_loss_cap_hit(101_000.0, 100_000.0, cap=0.02) is False  # up day


def test_loss_cap_never_false_triggers_without_baseline():
    assert dop.daily_loss_cap_hit(0.0, 100_000.0) is False       # unknown equity
    assert dop.daily_loss_cap_hit(50_000.0, 0.0) is False        # no prior close
    assert dop.daily_loss_cap_hit(-1.0, -1.0) is False
