"""Synthetic options backtester — Black‑Scholes over underlying OHLCV.

Honest limits (stated, not hidden): no skew/smile, no early exercise, vol
proxy = realized (understates rich IV regimes, so short‑premium results here
are CONSERVATIVE), fills at mid. Good for structure comparison and regime
sanity checks — not for absolute P&L claims.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    kind: str,
    rate: float = 0.04,
) -> float:
    """Black‑Scholes European price. At/past expiry returns intrinsic.

    Parameters
    ----------
    spot: float
        Current underlying price (>0).
    strike: float
        Option strike (>0).
    t_years: float
        Time to expiry in years.
    sigma: float
        Annualized volatility.
    kind: str
        Either ``"call"`` or ``"put"`` (case‑insensitive).
    rate: float, optional
        Continuously compounded risk‑free rate.

    Returns
    -------
    float
        Option price.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    kind = kind.lower()
    if kind not in {"call", "put"}:
        raise ValueError('kind must be "call" or "put"')
    intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    if t_years <= 0 or sigma <= 0:
        return intrinsic
    d1 = (
        math.log(spot / strike)
        + (rate + 0.5 * sigma**2) * t_years
    ) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    if kind == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * t_years) * _norm_cdf(d2)
    return strike * math.exp(-rate * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized close‑to‑close realized vol — the IV proxy."""
    rets = np.log(close / close.shift(1))
    return rets.rolling(window).std() * math.sqrt(TRADING_DAYS)


@dataclass
class SpreadLeg:
    kind: str          # call | put
    side: str          # buy | sell
    moneyness: float   # strike = moneyness * entry spot


@dataclass
class SpreadBacktestResult:
    trades: int
    wins: int
    total_pnl: float
    avg_pnl: float
    win_rate: float | None
    max_loss: float
    pnl_series: list[float]

    @property
    def summary(self) -> str:
        wr = f"{self.win_rate:.0%}" if self.win_rate is not None else "—"
        return (
            f"{self.trades} trades, win {wr}, total {self.total_pnl:+.2f}, "
            f"avg {self.avg_pnl:+.3f}, worst {self.max_loss:+.2f} "
            "(per 1x spread, mid fills)"
        )


def price_spread(
    spot: float,
    legs: list[SpreadLeg],
    strikes: list[float],
    t_years: float,
    sigma: float,
) -> float:
    """Signed value of the spread to its HOLDER (long premium positive)."""
    value = 0.0
    for leg, strike in zip(legs, strikes):
        p = bs_price(spot, strike, t_years, sigma, leg.kind)
        value += p if leg.side == "buy" else -p
    return value


def backtest_spread(
    df: pd.DataFrame,
    legs: list[SpreadLeg],
    entry_mask: pd.Series | None = None,
    dte: int = 35,
    hold_days: int = 21,
    vol_window: int = 20,
) -> SpreadBacktestResult:
    """Open the spread on each entry date, close by re‑pricing ``hold_days`` later.

    Parameters
    ----------
    df: pd.DataFrame
        Must contain a ``close`` column.
    legs: list[SpreadLeg]
        Specification of each leg.
    entry_mask: pd.Series | None, optional
        Boolean mask indicating entry dates; defaults to weekly entries.
    dte: int, optional
        Days to expiry at entry.
    hold_days: int, optional
        Holding period in days.
    vol_window: int, optional
        Window for realized volatility.

    Returns
    -------
    SpreadBacktestResult
        Aggregated backtest statistics.
    """
    close = df["close"].astype(float)
    vol = realized_vol(close, vol_window)

    if entry_mask is None:
        entry_mask = pd.Series(False, index=df.index)
        entry_mask.iloc[vol_window::5] = True

    pnls: list[float] = []
    n = len(df)

    for i in np.flatnonzero(entry_mask.to_numpy()):
        j = i + hold_days
        if j >= n:
            break
        sigma_in = float(vol.iloc[i]) if np.isfinite(vol.iloc[i]) else 0.0
        if sigma_in <= 0:
            continue

        spot_in, spot_out = float(close.iloc[i]), float(close.iloc[j])
        sigma_out = float(vol.iloc[j]) if np.isfinite(vol.iloc[j]) else sigma_in

        strikes = [leg.moneyness * spot_in for leg in legs]
        entry_v = price_spread(
            spot_in,
            legs,
            strikes,
            dte / TRADING_DAYS,
            sigma_in,
        )
        exit_v = price_spread(
            spot_out,
            legs,
            strikes,
            max(dte - hold_days, 0) / TRADING_DAYS,
            sigma_out,
        )
        pnls.append(exit_v - entry_v)

    wins = sum(1 for p in pnls if p > 0)
    return SpreadBacktestResult(
        trades=len(pnls),
        wins=wins,
        total_pnl=round(float(sum(pnls)), 4),
        avg_pnl=round(float(np.mean(pnls)), 4) if pnls else 0.0,
        win_rate=round(wins / len(pnls), 4) if pnls else None,
        max_loss=round(float(min(pnls)), 4) if pnls else 0.0,
        pnl_series=[round(float(p), 4) for p in pnls],
    )


# Ready‑made structures mirroring the Options desk's mleg specs
IRON_CONDOR = [
    SpreadLeg("put", "sell", 0.95),
    SpreadLeg("put", "buy", 0.91),
    SpreadLeg("call", "sell", 1.05),
    SpreadLeg("call", "buy", 1.09),
]
BULL_PUT_SPREAD = [SpreadLeg("put", "sell", 0.96), SpreadLeg("put", "buy", 0.92)]
BEAR_CALL_SPREAD = [SpreadLeg("call", "sell", 1.04), SpreadLeg("call", "buy", 1.08)]


# ----------------------------------------------------------------------
# Unit tests for edge‑case validation
# ----------------------------------------------------------------------
def test_bs_price_zero_time_returns_intrinsic():
    """When time to expiry is zero, price should equal intrinsic value."""
    spot = 100.0
    strike = 105.0
    intrinsic = max(spot - strike, 0.0)  # call intrinsic = 0
    price = bs_price(spot, strike, t_years=0.0, sigma=0.2, kind="call")
    assert price == intrinsic


def test_bs_price_invalid_kind_raises():
    """Invalid option kind should raise a ValueError."""
    try:
        bs_price(100, 100, 0.5, 0.2, kind="invalid")
    except ValueError as e:
        assert 'kind must be "call" or "put"' in str(e)
    else:
        assert False, "ValueError not raised for invalid kind"


def test_price_spread_empty_legs_returns_zero():
    """An empty leg list should result in a spread value of zero."""
    value = price_spread(spot=100.0, legs=[], strikes=[], t_years=0.1, sigma=0.2)
    assert value == 0.0


def test_backtest_spread_skips_nan_volatility():
    """When volatility is NaN (e.g., insufficient history), the entry should be skipped."""
    # Minimal DataFrame with only 3 rows; vol_window=5 ensures NaN volatility.
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    entry_mask = pd.Series([True, False, False], index=df.index)
    result = backtest_spread(df, legs=BULL_PUT_SPREAD, entry_mask=entry_mask, vol_window=5)
    # No trades should be recorded because volatility is NaN at the entry point.
    assert result.trades == 0
    assert result.pnl_series == []