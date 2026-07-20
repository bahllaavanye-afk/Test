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

TRADING_DAYS = 252

logger = logging.getLogger(__name__)


class OptionPricingError(Exception):
    """Base exception for option pricing related errors."""


class InvalidOptionKindError(OptionPricingError):
    """Raised when an unsupported option kind is supplied."""


class DataValidationError(OptionPricingError):
    """Raised when input data for backtesting is invalid."""


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
    """Black-Scholes European price. At/past expiry returns intrinsic."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if kind not in {"call", "put"}:
        raise InvalidOptionKindError(f"Unsupported option kind: {kind!r}")

    intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
    if t_years <= 0 or sigma <= 0:
        return intrinsic

    try:
        d1 = (
            math.log(spot / strike)
            + (rate + 0.5 * sigma**2) * t_years
        ) / (sigma * math.sqrt(t_years))
        d2 = d1 - sigma * math.sqrt(t_years)
    except Exception as exc:
        logger.error(
            "Error calculating d1/d2 in bs_price",
            extra={"spot": spot, "strike": strike, "t_years": t_years, "sigma": sigma, "kind": kind, "rate": rate},
        )
        raise OptionPricingError("Failed to compute Black-Scholes parameters") from exc

    if kind == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * t_years) * _norm_cdf(d2)
    return strike * math.exp(-rate * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized close-to-close realized vol — the IV proxy."""
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
            f"avg {self.avg_pnl:+.3f}, worst {self.max_loss:+.2f} (per 1x spread, mid fills)"
        )


def price_spread(
    spot: float,
    legs: list[SpreadLeg],
    strikes: list[float],
    t_years: float,
    sigma: float,
) -> float:
    """Signed value of the spread to its HOLDER (long premium positive)."""
    if len(legs) != len(strikes):
        raise OptionPricingError("Legs and strikes length mismatch")
    value = 0.0
    for leg, strike in zip(legs, strikes):
        try:
            p = bs_price(spot, strike, t_years, sigma, leg.kind)
        except OptionPricingError as exc:
            logger.error(
                "Failed to price leg in price_spread",
                extra={"spot": spot, "strike": strike, "t_years": t_years, "sigma": sigma, "leg": leg},
            )
            raise
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
    """Open the spread on each entry date, close by re-pricing hold_days later.

    entry_mask defaults to weekly entries (every 5th bar). P&L per unit spread:
    (exit value - entry value) — a net-credit structure enters at negative
    value, so decay toward zero is profit, exactly like the real book.
    """
    required_columns = {"close"}
    if not required_columns.issubset(df.columns):
        raise DataValidationError(f"DataFrame missing required columns: {required_columns - set(df.columns)}")

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

        try:
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
        except OptionPricingError as exc:
            logger.error(
                "Error pricing spread during backtest",
                extra={"entry_index": i, "exit_index": j, "spot_in": spot_in, "spot_out": spot_out},
            )
            continue

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


# Ready-made structures mirroring the Options desk's mleg specs
IRON_CONDOR = [
    SpreadLeg("put", "sell", 0.95),
    SpreadLeg("put", "buy", 0.91),
    SpreadLeg("call", "sell", 1.05),
    SpreadLeg("call", "buy", 1.09),
]
BULL_PUT_SPREAD = [SpreadLeg("put", "sell", 0.96), SpreadLeg("put", "buy", 0.92)]
BEAR_CALL_SPREAD = [SpreadLeg("call", "sell", 1.04), SpreadLeg("call", "buy", 1.08)]