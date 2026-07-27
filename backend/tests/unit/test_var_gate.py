"""The fifth documented risk check — diagrammed, implemented, never called.

`risk/CLAUDE.md` diagrams five gates inside `check_order()`:

    kelly · circuit_breaker · correlation · factor_exposure · var

`var.py` was fully implemented and `historical_var()` was called from nowhere.
This wires it in as "block if 1-day 99% VaR > 2% of NAV", the documented rule.

TWO THINGS MAKE THIS EASY TO GET WRONG, and both are pinned below.

1. FAIL-OPEN ON THIN DATA. `historical_var()` returns a *default* `var_99` of
   0.03 when it has fewer than 10 observations. Wired naively against a 2%
   limit, a cold start would block EVERY order until samples accumulate — a
   fleet-wide halt dressed as a risk control.

2. UNITS. `_risk_state_sync` polls equity every 60s, so returns built from
   every update are 1-MINUTE returns and a 99% VaR off those is ~30x too small
   against a 1-day limit. Equity is downsampled to
   `var_sample_interval_seconds` and scaled to one day by square-root-of-time.
   (sqrt-time assumes i.i.d. returns and understates tail risk under positive
   autocorrelation — it is a floor, not a ceiling.)

`factor_exposure.py` is the one remaining diagrammed gate still unwired: it
needs an aligned benchmark (SPY) return series at the same cadence as the
portfolio series, which nothing currently produces. Left out rather than fed a
mismatched series — see test_factor_exposure_is_still_honestly_unwired.
"""
from __future__ import annotations

import math

import pytest

from app.brokers.base import OrderRequest
from app.risk.manager import RiskManager


def _order(qty=1.0, price=100.0, bucket="directional", symbol="AAPL") -> OrderRequest:
    return OrderRequest(
        symbol=symbol, side="buy", order_type="limit",
        quantity=qty, limit_price=price, risk_bucket=bucket,
    )


def _rm(**kw) -> RiskManager:
    kw.setdefault("initial_equity", 100_000)
    kw.setdefault("var_sample_interval_seconds", 0.0)   # sample every update
    return RiskManager(**kw)


def _feed_equity(rm: RiskManager, values) -> None:
    for v in values:
        rm.update_equity(float(v))


# ── fail-open: the trap that would halt the fleet ────────────────────────────

@pytest.mark.asyncio
async def test_a_cold_start_does_not_block_every_order():
    """historical_var() returns a DEFAULT var_99=0.03 below 10 observations.

    Against a 2% limit that default alone would reject everything. The gate
    must decline to have an opinion instead.
    """
    rm = _rm()
    rm.update_equity(100_000)
    decision = await rm.check_order(_order())
    assert decision.allowed, (
        "a cold start has no VaR estimate; blocking on the library's "
        "insufficient-data default is a fleet-wide halt, not a risk control"
    )
    assert "VaR" not in decision.reason


@pytest.mark.asyncio
async def test_just_under_the_sample_minimum_still_does_not_block():
    rm = _rm(min_samples_for_var=20)
    _feed_equity(rm, [100_000 * (1 + 0.001 * i) for i in range(19)])
    assert len(rm._equity_samples) < rm.min_samples_for_var
    assert (await rm.check_order(_order())).allowed


def test_the_insufficient_data_sentinel_is_checked_not_just_the_count():
    """Belt and braces: both the sample count AND the returned method."""
    rm = _rm(min_samples_for_var=2)
    _feed_equity(rm, [100_000, 100_100, 100_200])   # 2 returns → var.py default
    assert rm._var_decision() is None, (
        "method == 'default_insufficient_data' must be treated as no opinion"
    )


# ── it actually blocks when the risk is real ─────────────────────────────────

@pytest.mark.asyncio
async def test_a_violently_volatile_book_is_blocked():
    rm = _rm(max_var_pct=0.02, var_sample_interval_seconds=3600.0,
             min_samples_for_var=20)
    # ±6% hourly swings — a 1-day 99% VaR far beyond 2%.
    equity, values = 100_000.0, []
    for i in range(40):
        equity *= 1.06 if i % 2 else 0.94
        values.append(equity)
    rm._equity_samples.extend(values)          # bypass the interval throttle
    rm._equity = values[-1]

    decision = await rm.check_order(_order())
    assert not decision.allowed
    assert "VaR" in decision.reason and "exceeds limit" in decision.reason


@pytest.mark.asyncio
async def test_a_calm_book_is_not_blocked():
    rm = _rm(max_var_pct=0.02, var_sample_interval_seconds=3600.0,
             min_samples_for_var=20)
    equity, values = 100_000.0, []
    for i in range(40):
        equity *= 1.0002 if i % 2 else 0.9999   # ~2bp hourly noise
        values.append(equity)
    rm._equity_samples.extend(values)
    rm._equity = values[-1]

    decision = await rm.check_order(_order())
    assert decision.allowed, f"calm book should pass, got: {decision.reason}"


# ── units ────────────────────────────────────────────────────────────────────

def test_var_is_scaled_from_the_sample_horizon_to_one_day():
    """A 1-minute VaR compared against a 1-day limit is ~30x too permissive."""
    rm = _rm(var_sample_interval_seconds=3600.0, min_samples_for_var=20)
    equity, values = 100_000.0, []
    for i in range(40):
        equity *= 1.01 if i % 2 else 0.99
        values.append(equity)
    rm._equity_samples.extend(values)
    rm._equity = values[-1]

    rm._var_decision()
    reported = rm.last_var
    assert reported, "the gate must publish what it computed"

    expected_scale = math.sqrt(24.0)            # hourly → daily
    assert reported["var_99_daily"] == pytest.approx(
        reported["var_99_per_sample"] * expected_scale, rel=1e-3
    ), "hourly VaR must be scaled by sqrt(24), not compared raw"


def test_a_finer_sample_interval_scales_by_more():
    """The scaling must follow the configured interval, not a constant."""
    def _daily_for(interval: float) -> float:
        rm = _rm(var_sample_interval_seconds=interval, min_samples_for_var=20)
        equity, values = 100_000.0, []
        for i in range(40):
            equity *= 1.01 if i % 2 else 0.99
            values.append(equity)
        rm._equity_samples.extend(values)
        rm._equity = values[-1]
        rm._var_decision()
        return rm.last_var["var_99_daily"]

    hourly = _daily_for(3600.0)
    minutely = _daily_for(60.0)
    assert minutely > hourly, (
        "60s samples cover 1440 periods/day vs 24 — the same per-sample VaR "
        "must annualise to a larger daily figure"
    )
    assert minutely / hourly == pytest.approx(math.sqrt(1440 / 24), rel=1e-3)


# ── sampling hygiene ─────────────────────────────────────────────────────────

def test_equity_is_downsampled_not_recorded_on_every_poll():
    rm = _rm(var_sample_interval_seconds=3600.0)
    for i in range(50):
        rm.update_equity(100_000 + i)
    assert len(rm._equity_samples) == 1, (
        "60s polls must not become the VaR series; that is a 1-minute VaR"
    )


def test_a_clamped_zero_equity_is_not_recorded_as_an_observation():
    """main.py clamps negative broker equity to 0 to trip the halt.

    Recording that as a sample would inject a -100% return and poison the VaR
    window for as long as it stays in it.
    """
    rm = _rm()
    _feed_equity(rm, [100_000, 101_000, 99_000])
    before = list(rm._equity_samples)
    rm.update_equity(0.0)
    assert list(rm._equity_samples) == before


def test_the_sample_window_is_bounded():
    rm = _rm(min_samples_for_var=20)
    for i in range(10_000):
        rm.update_equity(100_000 + i)
    assert len(rm._equity_samples) <= rm._equity_samples.maxlen


def test_var_failure_never_breaks_the_gate(monkeypatch):
    """This sits in front of every order — it must not raise."""
    import app.risk.manager as mgr

    rm = _rm(min_samples_for_var=5)
    _feed_equity(rm, [100_000 * (1 + 0.01 * i) for i in range(30)])
    monkeypatch.setattr(mgr, "historical_var", lambda *a, **kw: 1 / 0)
    assert rm._var_decision() is None, "a VaR error must mean no opinion, not a crash"


# ── the remaining unwired gate, stated honestly ──────────────────────────────

def test_factor_exposure_is_still_honestly_unwired():
    """Four of the five diagrammed gates now run. This one does not.

    compute_factor_exposure() needs portfolio returns AND an aligned benchmark
    (SPY) series at the same cadence. The portfolio series comes from hourly
    NAV samples; marks are sampled on a different schedule and SPY is not
    guaranteed to be in the universe at all. Feeding it a misaligned series
    would produce a confident, wrong beta — worse than no beta.

    This test exists so the gap stays visible rather than being quietly
    forgotten, which is exactly how all five ended up unwired.
    """
    import app.risk.manager as mgr

    assert not hasattr(mgr, "compute_factor_exposure"), (
        "if factor exposure is now wired, replace this test with real coverage "
        "of the beta limit rather than deleting it"
    )
