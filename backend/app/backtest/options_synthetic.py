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
    *,
    max_vol: float | None = None,
    pnl_target_factor: float = 0.5,
    pnl_stop_factor: float = -0.5,
) -> SpreadBacktestResult:
    """Open the spread on each entry date, close by re‑pricing ``hold_days`` later.

    The entry logic is tightened with optional volatility filtering and a
    confirmation that the spread is a credit (positive entry value).  Exit
    logic now includes early‑exit based on profit‑target or stop‑loss
    thresholds expressed as fractions of the initial credit.

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
        Maximum holding period in days.
    vol_window: int, optional
        Window for realized volatility.
    max_vol: float | None, optional
        Upper bound on realized volatility for an entry to be considered.
    pnl_target_factor: float, optional
        Profit‑target as a fraction of the entry credit (positive).
    pnl_stop_factor: float, optional
        Stop‑loss as a fraction of the entry credit (negative).

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
        # Basic bounds check
        if i + hold_days >= n:
            break

        sigma_in = float(vol.iloc[i]) if np.isfinite(vol.iloc[i]) else 0.0
        if sigma_in <= 0:
            continue
        if max_vol is not None and sigma_in > max_vol:
            continue

        spot_in = float(close.iloc[i])
        strikes = [leg.moneyness * spot_in for leg in legs]

        entry_v = price_spread(
            spot_in,
            legs,
            strikes,
            dte / TRADING_DAYS,
            sigma_in,
        )
        # Require a credit (positive entry value) for credit spreads
        if entry_v <= 0:
            continue

        # Define dynamic exit thresholds based on entry credit
        target = entry_v * pnl_target_factor
        stop = entry_v * pnl_stop_factor

        exit_pnl: float | None = None
        # Walk forward day‑by‑day to allow early exit
        for offset in range(1, hold_days + 1):
            j = i + offset
            if j >= n:
                break
            spot_j = float(close.iloc[j])
            sigma_j = float(vol.iloc[j]) if np.isfinite(vol.iloc[j]) else sigma_in
            remaining_t = max(dte - offset, 0) / TRADING_DAYS
            exit_v = price_spread(
                spot_j,
                legs,
                strikes,
                remaining_t,
                sigma_j,
            )
            pnl = exit_v - entry_v
            if pnl >= target or pnl <= stop:
                exit_pnl = pnl
                break

        # If no early exit triggered, use final day value
        if exit_pnl is None:
            spot_out = float(close.iloc[i + hold_days])
            sigma_out = float(vol.iloc[i + hold_days]) if np.isfinite(vol.iloc[i + hold_days]) else sigma_in
            exit_v = price_spread(
                spot_out,
                legs,
                strikes,
                max(dte - hold_days, 0) / TRADING_DAYS,
                sigma_out,
            )
            exit_pnl = exit_v - entry_v

        pnls.append(exit_pnl)

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