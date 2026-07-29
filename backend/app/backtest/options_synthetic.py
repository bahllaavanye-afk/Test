"""Synthetic options backtester — Black‑Scholes over underlying OHLCV.

This module provides a lightweight, deterministic backtester for option spreads
using the Black‑Scholes formula. It assumes no volatility skew or smile, no
early exercise, and uses realized close‑to‑close volatility as a proxy for the
implied volatility. The implementation is intended for structural comparisons
and regime sanity checks rather than absolute P&L claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function.

    Parameters
    ----------
    x: float
        Value at which to evaluate the CDF.

    Returns
    -------
    float
        The probability that a standard normal variable is less than or equal to ``x``.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    kind: str,
    rate: float = 0.04,
) -> float:
    """Black‑Scholes European option price; returns intrinsic value if expired.

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
        The option price.
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
    """Compute annualized close‑to‑close realized volatility.

    Parameters
    ----------
    close: pd.Series
        Series of closing prices.
    window: int, optional
        Rolling window size (in days) for volatility calculation.

    Returns
    -------
    pd.Series
        Realized volatility series aligned with ``close``.
    """
    rets = np.log(close / close.shift(1))
    return rets.rolling(window).std() * math.sqrt(TRADING_DAYS)


@dataclass
class SpreadLeg:
    """Specification of a single leg in an option spread.

    Attributes
    ----------
    kind: str
        Option type, either ``"call"`` or ``"put"``.
    side: str
        Position side, ``"buy"`` for long or ``"sell"`` for short.
    moneyness: float
        Multiplier applied to the entry spot to obtain the strike price.
    """
    kind: str          # call | put
    side: str          # buy | sell
    moneyness: float   # strike = moneyness * entry spot


@dataclass
class SpreadBacktestResult:
    """Aggregated results from a spread backtest.

    Attributes
    ----------
    trades: int
        Total number of executed trades.
    wins: int
        Number of trades with positive P&L.
    total_pnl: float
        Sum of all trade P&L values.
    avg_pnl: float
        Mean P&L per trade.
    win_rate: float | None
        Proportion of winning trades; ``None`` if no trades were executed.
    max_loss: float
        Largest negative P&L (worst loss).
    pnl_series: list[float]
        List of individual trade P&L values.
    """
    trades: int
    wins: int
    total_pnl: float
    avg_pnl: float
    win_rate: float | None
    max_loss: float
    pnl_series: list[float]

    @property
    def summary(self) -> str:
        """Human‑readable summary of the backtest statistics."""
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
    """Calculate the signed value of an option spread for its holder.

    Parameters
    ----------
    spot: float
        Underlying price at valuation.
    legs: list[SpreadLeg]
        List describing each leg of the spread.
    strikes: list[float]
        Corresponding strike prices for each leg.
    t_years: float
        Time to expiry (in years) for pricing.
    sigma: float
        Annualized volatility used in Black‑Scholes pricing.

    Returns
    -------
    float
        Net value of the spread (positive for a long holder).
    """
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
    """Backtest an option spread by opening on entry dates and closing after a holding period.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing at least a ``close`` column with price data.
    legs: list[SpreadLeg]
        Specification of each leg in the spread.
    entry_mask: pd.Series | None, optional
        Boolean mask indicating entry dates; defaults to a weekly schedule.
    dte: int, optional
        Days to expiry at entry.
    hold_days: int, optional
        Number of days to hold the spread before exiting.
    vol_window: int, optional
        Window size for realized volatility estimation.

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