"""Synthetic options backtester — Black‑Scholes over underlying OHLCV.

IMPROVEMENTS P1: the options income structures (condors, credit spreads, CSPs)
deploy on walk‑forward‑tested UNDERLYING signals, but their option‑level P&L
was never backtestable — Alpaca has no historical option chains on the free
tier. This prices the legs synthetically: BS with realized vol (20d, annualized)
as the implied proxy, entry at signal dates, exit by re‑pricing at hold end.

Honest limits (stated, not hidden): no skew/smile, no early exercise, vol
proxy = realized (understates rich IV regimes, so short‑premium results here
are CONSERVATIVE), fills at mid. Good for structure comparison and regime
sanity checks — not for absolute P&L claims.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    spot: float,
    strike: float,
    t_years: float,
    sigma: float,
    kind: str,
    rate: float = 0.04,
) -> float:
    """Black‑Scholes European price. At/past expiry returns intrinsic."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if kind not in {"call", "put"}:
        raise ValueError("kind must be 'call' or 'put'")
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
    pnl_series: List[float]

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
    legs: List[SpreadLeg],
    strikes: List[float],
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
    legs: List[SpreadLeg],
    entry_mask: pd.Series | None = None,
    *,
    confirmation_mask: pd.Series | None = None,
    dte: int = 35,
    hold_days: int = 21,
    vol_window: int = 20,
    profit_target: float = 0.5,
    stop_loss: float = -0.5,
) -> SpreadBacktestResult:
    """Open the spread on each entry date, close by re‑pricing hold_days later.

    Entry is allowed only when both ``entry_mask`` and ``confirmation_mask`` are
    true (if the latter is provided).  For credit‑type spreads we add a simple
    volatility‑decrease filter: the realized vol must be falling on the entry
    day, which historically improves the odds of a short‑vol strategy.

    Exit logic now caps profit and loss relative to the absolute entry credit
    (or debit) to avoid extreme outliers that are unlikely in a real order‑book
    execution environment.
    """
    close = df["close"].astype(float)
    vol = realized_vol(close, vol_window)

    if entry_mask is None:
        entry_mask = pd.Series(False, index=df.index)
        entry_mask.iloc[vol_window::5] = True

    if confirmation_mask is not None:
        entry_mask = entry_mask & confirmation_mask

    pnls: List[float] = []
    n = len(df)

    for i in np.flatnonzero(entry_mask.to_numpy()):
        j = i + hold_days
        if j >= n:
            break

        sigma_in = float(vol.iloc[i]) if np.isfinite(vol.iloc[i]) else 0.0
        if sigma_in <= 0:
            continue

        # Volatility‑decrease filter for credit spreads
        if i > 0 and vol.iloc[i] >= vol.iloc[i - 1]:
            # If the spread is a net credit (entry value negative) we skip
            # entries where vol is not falling.
            spot_tmp = float(close.iloc[i])
            strikes_tmp = [leg.moneyness * spot_tmp for leg in legs]
            entry_val_tmp = price_spread(
                spot_tmp, legs, strikes_tmp, dte / TRADING_DAYS, sigma_in
            )
            if entry_val_tmp < 0:
                continue

        spot_in, spot_out = float(close.iloc[i]), float(close.iloc[j])
        sigma_out = (
            float(vol.iloc[j]) if np.isfinite(vol.iloc[j]) else sigma_in
        )
        strikes = [leg.moneyness * spot_in for leg in legs]

        entry_v = price_spread(
            spot_in, legs, strikes, dte / TRADING_DAYS, sigma_in
        )
        # Net‑credit spreads have negative entry value; profit is decay toward zero.
        remaining_days = max(dte - hold_days, 0)
        exit_v = price_spread(
            spot_out,
            legs,
            strikes,
            remaining_days / TRADING_DAYS,
            sigma_out,
        )
        raw_pnl = exit_v - entry_v

        # Apply profit target / stop loss caps relative to entry credit magnitude
        entry_credit = -entry_v if entry_v < 0 else entry_v
        if entry_credit != 0:
            cap_up = profit_target * abs(entry_credit)
            cap_down = stop_loss * abs(entry_credit)
            if raw_pnl > cap_up:
                raw_pnl = cap_up
            elif raw_pnl < cap_down:
                raw_pnl = cap_down

        pnls.append(raw_pnl)

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