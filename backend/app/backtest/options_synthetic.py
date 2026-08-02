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
    """Standard normal cumulative distribution function (scalar)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_cdf_np(x: np.ndarray) -> np.ndarray:
    """Standard normal cumulative distribution function (vectorized)."""
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


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
    kind = kind.lower()
    if kind not in {"call", "put"}:
        raise ValueError('kind must be "call" or "put"')
    intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    if t_years <= 0 or sigma <= 0:
        return intrinsic
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * t_years) / (
        sigma * math.sqrt(t_years)
    )
    d2 = d1 - sigma * math.sqrt(t_years)
    if kind == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * t_years) * _norm_cdf(d2)
    return strike * math.exp(-rate * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_price_np(
    spot: np.ndarray,
    strike: np.ndarray,
    t_years: float,
    sigma: np.ndarray,
    kind: str,
    rate: float = 0.04,
) -> np.ndarray:
    """Vectorized Black‑Scholes price for arrays of spot, strike, sigma."""
    kind = kind.lower()
    if kind not in {"call", "put"}:
        raise ValueError('kind must be "call" or "put"')
    intrinsic = np.where(
        kind == "call",
        np.maximum(spot - strike, 0.0),
        np.maximum(strike - spot, 0.0),
    )
    # Mask where option is alive
    alive = (t_years > 0) & (sigma > 0)
    result = intrinsic.copy()
    if not np.any(alive):
        return result
    sqrt_t = math.sqrt(t_years)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if kind == "call":
        price = spot * _norm_cdf_np(d1) - strike * math.exp(-rate * t_years) * _norm_cdf_np(d2)
    else:
        price = strike * math.exp(-rate * t_years) * _norm_cdf_np(-d2) - spot * _norm_cdf_np(-d1)
    result[alive] = price[alive]
    return result


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


def price_spread_vectorized(
    spot: np.ndarray,
    legs: list[SpreadLeg],
    strikes: np.ndarray,
    t_years: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Vectorized signed value of the spread to its HOLDER (long premium positive)."""
    value = np.zeros_like(spot, dtype=float)
    for leg in legs:
        price = bs_price_np(spot, strikes, t_years, sigma, leg.kind)
        if leg.side == "buy":
            value += price
        else:
            value -= price
    return value


def backtest_spread(
    df: pd.DataFrame,
    legs: list[SpreadLeg],
    entry_mask: pd.Series | None = None,
    dte: int = 35,
    hold_days: int = 21,
    vol_window: int = 20,
) -> SpreadBacktestResult:
    """Open the spread on each entry date, close by re‑pricing ``hold_days`` later."""
    close = df["close"].astype(float)
    vol = realized_vol(close, vol_window)

    if entry_mask is None:
        entry_mask = pd.Series(False, index=df.index)
        entry_mask.iloc[vol_window::5] = True

    idx_entry = np.flatnonzero(entry_mask.to_numpy())
    n = len(df)
    idx_exit = idx_entry + hold_days
    valid = idx_exit < n
    idx_entry = idx_entry[valid]
    idx_exit = idx_exit[valid]

    sigma_in = vol.iloc[idx_entry].to_numpy()
    sigma_in = np.where(np.isfinite(sigma_in), sigma_in, 0.0)

    # Filter out non‑positive vol entries early
    alive_mask = sigma_in > 0
    idx_entry = idx_entry[alive_mask]
    idx_exit = idx_exit[alive_mask]
    sigma_in = sigma_in[alive_mask]

    if len(idx_entry) == 0:
        return SpreadBacktestResult(
            trades=0,
            wins=0,
            total_pnl=0.0,
            avg_pnl=0.0,
            win_rate=None,
            max_loss=0.0,
            pnl_series=[],
        )

    spot_in = close.iloc[idx_entry].to_numpy()
    spot_out = close.iloc[idx_exit].to_numpy()

    sigma_out = vol.iloc[idx_exit].to_numpy()
    sigma_out = np.where(np.isfinite(sigma_out), sigma_out, sigma_in)

    # Strikes are based on entry spot only (as in original implementation)
    strikes = np.array([leg.moneyness * spot_in for leg in legs], dtype=float).T

    # Compute entry and exit values vectorized
    entry_v = price_spread_vectorized(
        spot_in,
        legs,
        strikes,
        dte / TRADING_DAYS,
        sigma_in,
    )
    exit_v = price_spread_vectorized(
        spot_out,
        legs,
        strikes,
        max(dte - hold_days, 0) / TRADING_DAYS,
        sigma_out,
    )
    pnls = exit_v - entry_v

    wins = int(np.sum(pnls > 0))
    total_pnl = float(np.sum(pnls))
    avg_pnl = float(np.mean(pnls)) if pnls.size else 0.0
    max_loss = float(np.min(pnls)) if pnls.size else 0.0
    win_rate = round(wins / pnls.size, 4) if pnls.size else None

    return SpreadBacktestResult(
        trades=int(pnls.size),
        wins=wins,
        total_pnl=round(total_pnl, 4),
        avg_pnl=round(avg_pnl, 4),
        win_rate=win_rate,
        max_loss=round(max_loss, 4),
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