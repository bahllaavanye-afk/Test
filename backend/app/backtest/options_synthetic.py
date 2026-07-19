"""Synthetic options backtester — Black-Scholes over underlying OHLCV.

IMPROVEMENTS P1: the options income structures (condors, credit spreads, CSPs)
deploy on walk-forward-tested UNDERLYING signals, but their option-level P&L
was never backtestable — Alpaca has no historical option chains on the free
tier. This prices the legs synthetically: BS with realized vol (20d, annualized)
as the implied proxy, entry at signal dates, exit by re-pricing at hold end.

Honest limits (stated, not hidden): no skew/smile, no early exercise, vol
proxy = realized (understates rich IV regimes, so short-premium results here
are CONSERVATIVE), fills at mid. Good for structure comparison and regime
sanity checks — not for absolute P&L claims.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Logging configuration (structured)
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class OptionPricingError(ValueError):
    """Raised when inputs to the Black‑Scholes pricing function are invalid."""


class SpreadPricingError(RuntimeError):
    """Raised when constructing or pricing a spread fails due to invalid leg data."""


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for a standard normal distribution."""
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
        Time to expiry in years (>=0). Zero or negative yields intrinsic.
    sigma: float
        Volatility (annualized, >0). Zero or negative yields intrinsic.
    kind: str
        'call' or 'put'.
    rate: float, optional
        Risk‑free rate, default 0.04.

    Raises
    ------
    OptionPricingError
        If `kind` is not one of {'call', 'put'} or if spot/strike are non‑positive.
    """
    if spot <= 0 or strike <= 0:
        raise OptionPricingError("spot and strike must be positive")
    if kind not in {"call", "put"}:
        raise OptionPricingError(f"invalid option kind '{kind}'; expected 'call' or 'put'")

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


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized close‑to‑close realized vol — the IV proxy."""
    rets = np.log(close / close.shift(1))
    return rets.rolling(window).std() * math.sqrt(TRADING_DAYS)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
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
            f"avg {self.avg_pnl:+.3f}, worst {self.max_loss:+.2f} (per 1x spread, mid fills)"
        )


# --------------------------------------------------------------------------- #
# Core pricing utilities
# --------------------------------------------------------------------------- #
def _validate_leg(leg: SpreadLeg) -> None:
    """Validate a single spread leg. Raises SpreadPricingError on failure."""
    if leg.kind not in {"call", "put"}:
        raise SpreadPricingError(f"invalid leg kind '{leg.kind}'; expected 'call' or 'put'")
    if leg.side not in {"buy", "sell"}:
        raise SpreadPricingError(f"invalid leg side '{leg.side}'; expected 'buy' or 'sell'")
    if leg.moneyness <= 0:
        raise SpreadPricingError("leg moneyness must be positive")


def price_spread(
    spot: float, legs: list[SpreadLeg], strikes: list[float], t_years: float, sigma: float
) -> float:
    """Signed value of the spread to its HOLDER (long premium positive)."""
    if len(legs) != len(strikes):
        raise SpreadPricingError(
            f"legs count ({len(legs)}) does not match strikes count ({len(strikes)})"
        )
    value = 0.0
    for leg, strike in zip(legs, strikes):
        try:
            _validate_leg(leg)
            p = bs_price(spot, strike, t_years, sigma, leg.kind)
            value += p if leg.side == "buy" else -p
        except (OptionPricingError, SpreadPricingError) as exc:
            logger.error(
                "Failed to price leg %s with strike %.4f: %s",
                leg,
                strike,
                exc,
                exc_info=True,
            )
            raise
    return value


# --------------------------------------------------------------------------- #
# Backtesting routine
# --------------------------------------------------------------------------- #
def backtest_spread(
    df: pd.DataFrame,
    legs: list[SpreadLeg],
    entry_mask: pd.Series | None = None,
    dte: int = 35,
    hold_days: int = 21,
    vol_window: int = 20,
) -> SpreadBacktestResult:
    """Open the spread on each entry date, close by re‑pricing hold_days later.

    Parameters
    ----------
    df: pd.DataFrame
        Must contain a 'close' column with numeric values.
    legs: list[SpreadLeg]
        Specification of the spread legs.
    entry_mask: pd.Series | None
        Boolean mask indicating entry dates. If None, defaults to weekly entries.
    dte: int
        Days to expiry at entry.
    hold_days: int
        Holding period in days.
    vol_window: int
        Rolling window for realized volatility.

    Returns
    -------
    SpreadBacktestResult
        Aggregated performance metrics.
    """
    # Basic input validation
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")
    if not legs:
        raise ValueError("At least one SpreadLeg must be provided")
    for leg in legs:
        _validate_leg(leg)

    close = df["close"].astype(float)
    vol = realized_vol(close, vol_window)

    if entry_mask is None:
        entry_mask = pd.Series(False, index=df.index)
        entry_mask.iloc[vol_window::5] = True
    else:
        if not isinstance(entry_mask, pd.Series):
            raise TypeError("entry_mask must be a pandas Series")
        if entry_mask.dtype != bool:
            entry_mask = entry_mask.astype(bool)

    pnls: list[float] = []
    n = len(df)

    for i in np.flatnonzero(entry_mask.to_numpy()):
        j = i + hold_days
        if j >= n:
            break
        try:
            sigma_in = float(vol.iloc[i]) if np.isfinite(vol.iloc[i]) else 0.0
            if sigma_in <= 0:
                continue
            spot_in, spot_out = float(close.iloc[i]), float(close.iloc[j])
            sigma_out = (
                float(vol.iloc[j]) if np.isfinite(vol.iloc[j]) else sigma_in
            )
            strikes = [leg.moneyness * spot_in for leg in legs]
            entry_v = price_spread(
                spot_in, legs, strikes, dte / TRADING_DAYS, sigma_in
            )
            exit_v = price_spread(
                spot_out,
                legs,
                strikes,
                max(dte - hold_days, 0) / TRADING_DAYS,
                sigma_out,
            )
            pnls.append(exit_v - entry_v)
        except Exception as exc:
            logger.error(
                "Error processing trade starting at index %d: %s", i, exc, exc_info=True
            )
            continue

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


# --------------------------------------------------------------------------- #
# Ready‑made structures mirroring the Options desk's mleg specs
# --------------------------------------------------------------------------- #
IRON_CONDOR = [
    SpreadLeg("put", "sell", 0.95),
    SpreadLeg("put", "buy", 0.91),
    SpreadLeg("call", "sell", 1.05),
    SpreadLeg("call", "buy", 1.09),
]
BULL_PUT_SPREAD = [SpreadLeg("put", "sell", 0.96), SpreadLeg("put", "buy", 0.92)]
BEAR_CALL_SPREAD = [
    SpreadLeg("call", "sell", 1.04),
    SpreadLeg("call", "buy", 1.08),
]