from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #

# Default parameters
DEFAULT_INITIAL_EQUITY: float = 100_000.0
DEFAULT_COMMISSION_PCT: float = 0.001
DEFAULT_SLIPPAGE_PCT: float = 0.0005
DEFAULT_RISK_FREE_ANNUAL: float = 0.05

# Model / calculation constants
TRADING_DAYS_PER_YEAR: int = 252
MAX_SLIPPAGE_MULTIPLIER: float = 5.0
VOLUME_CLIP_LOWER: float = 1.0
PARTICIPATION_CLIP_LOWER: float = 0.0
PARTICIPATION_CLIP_UPPER: float = 1.0
OMEGA_THRESHOLD: float = 0.0
EPSILON: float = 1e-10

# Error message constants
ERR_SIGNALS_TYPE = "signals must be a pandas Series"
ERR_PRICES_TYPE = "prices must be a pandas Series"
ERR_SIGNALS_EMPTY = "signals series is empty"
ERR_PRICES_EMPTY = "prices series is empty"
ERR_INDEX_MISMATCH = "signals and prices must share the same index"
ERR_OPENS_TYPE = "opens must be a pandas Series when provided"
ERR_OPENS_INDEX = "opens must share the same index as prices"
ERR_VOLUME_TYPE = "volume must be a pandas Series when provided"
ERR_VOLUME_INDEX = "volume must share the same index as prices"


@dataclass
class BacktestMetrics:
    # Returns
    total_return: float
    annualized_return: float

    # Risk-adjusted
    sharpe: float
    sortino: float
    calmar: float
    omega_ratio: float      # prob-weighted gains / prob-weighted losses above threshold
    ulcer_index: float      # RMS of drawdown depth — penalises prolonged underwater periods

    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration_days: int

    # Trading stats
    num_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy: float       # avg_win * win_rate − avg_loss * (1 − win_rate)

    # Equity curve (for charting)
    equity_curve: list[dict] = field(default_factory=list)


def _omega_ratio(returns: np.ndarray, threshold: float = OMEGA_THRESHOLD) -> float:
    """Omega ratio: sum(gains above threshold) / sum(losses below threshold)."""
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    if losses.sum() == 0:
        return float("inf")
    return float(gains.sum() / losses.sum())


def _ulcer_index(equity: np.ndarray) -> float:
    """Ulcer Index: RMS of percentage drawdown. Higher = more painful."""
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / peak * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def _adaptive_slippage(
    trade_size_usd: pd.Series,
    volume_usd: pd.Series | None,
    base_slippage_pct: float,
) -> pd.Series:
    """
    Scale slippage with participation rate using Kyle's sqrt-of-volume model.

    slippage = base * sqrt(participation_rate)
    where participation_rate = trade_size / (price * daily_volume).

    Caps at MAX_SLIPPAGE_MULTIPLIER × base to avoid extreme values on illiquid days.
    """
    if volume_usd is None or (volume_usd == 0).all():
        return pd.Series(base_slippage_pct, index=trade_size_usd.index)

    participation = (trade_size_usd / volume_usd.clip(lower=VOLUME_CLIP_LOWER)).clip(
        PARTICIPATION_CLIP_LOWER, PARTICIPATION_CLIP_UPPER
    )
    scaled = base_slippage_pct * np.sqrt(participation)
    return scaled.clip(upper=base_slippage_pct * MAX_SLIPPAGE_MULTIPLIER)


def run_backtest(
    signals: pd.Series,
    prices: pd.Series,
    opens: pd.Series | None = None,
    volume: pd.Series | None = None,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    fill_at_open: bool = True,
    risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL,
) -> BacktestMetrics:
    """
    Vectorized backtest.

    Parameters
    ----------
    signals     : +1 buy, -1 sell, 0 hold. Must be pre-shifted to avoid lookahead.
    prices      : OHLCV close prices (used for mark-to-market).
    opens       : Open prices. When fill_at_open=True, trades execute at next open.
                  Falls back to close if not provided.
    volume      : Daily volume (shares or contracts). Used for adaptive slippage.
                  Pass None to use flat slippage_pct.
    fill_at_open: If True, position changes fill at the bar's OPEN price, not
                  the previous bar's close. This is more realistic for EOD signals.
    """
    # --------------------------------------------------------------------- #
    # Input validation
    # --------------------------------------------------------------------- #
    try:
        if not isinstance(signals, pd.Series):
            raise TypeError(ERR_SIGNALS_TYPE)
        if not isinstance(prices, pd.Series):
            raise TypeError(ERR_PRICES_TYPE)
        if signals.empty:
            raise ValueError(ERR_SIGNALS_EMPTY)
        if prices.empty:
            raise ValueError(ERR_PRICES_EMPTY)
        if not signals.index.equals(prices.index):
            raise ValueError(ERR_INDEX_MISMATCH)

        if opens is not None:
            if not isinstance(opens, pd.Series):
                raise TypeError(ERR_OPENS_TYPE)
            if not opens.index.equals(prices.index):
                raise ValueError(ERR_OPENS_INDEX)
        if volume is not None:
            if not isinstance(volume, pd.Series):
                raise TypeError(ERR_VOLUME_TYPE)
            if not volume.index.equals(prices.index):
                raise ValueError(ERR_VOLUME_INDEX)
    except (TypeError, ValueError) as e:
        logger.error("Input validation error: %s", e, exc_info=True)
        raise

    # --------------------------------------------------------------------- #
    # Core computation wrapped to capture unexpected failures
    # --------------------------------------------------------------------- #
    try:
        fill_prices = opens if (fill_at_open and opens is not None) else prices

        df = pd.DataFrame(
            {
                "signal": signals,
                "price": prices,
                "fill_price": fill_prices,
            }
        ).dropna(subset=["signal", "price"])

        # Build optional volume column outside the main DataFrame to avoid dtype confusion
        _volume_usd: pd.Series | None = None
        if volume is not None:
            _volume_usd = volume.reindex(df.index).fillna(0) * df["price"]

        # Carry forward last signal to maintain position
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        # Shift so position change takes effect at *next* bar's open
        df["position"] = df["position"].shift(1).fillna(0)

        # Detect transitions (direction changes or new entries)
        df["trade"] = df["position"].diff().fillna(0)
        trade_mask = df["trade"] != 0

        # Volume-adaptive slippage on transition bars only
        trade_size_usd = df["trade"].abs() * df["fill_price"] * initial_equity / df["fill_price"].iloc[0]
        slip = _adaptive_slippage(trade_size_usd, _volume_usd, slippage_pct)

        total_cost_pct = (commission_pct + slip) * trade_mask.astype(float)

        # Daily P&L: mark-to-market returns on the held position
        df["bar_return"] = df["price"].pct_change().fillna(0)
        df["pnl"] = df["position"] * df["bar_return"] - total_cost_pct

        df["equity"] = initial_equity * (1 + df["pnl"]).cumprod()
        df["equity"] = df["equity"].ffill().fillna(initial_equity)

        equity = df["equity"].values
        returns = df["pnl"].values
        rf_daily = risk_free_annual / TRADING_DAYS_PER_YEAR

        # ── Sharpe ────────────────────────────────────────────────────────────────
        excess = returns - rf_daily
        _excess_std = float(np.std(excess))
        sharpe = (
            float(excess.mean() / _excess_std * np.sqrt(TRADING_DAYS_PER_YEAR))
            if _excess_std > EPSILON
            else 0.0
        )

        # ── Sortino ───────────────────────────────────────────────────────────────
        downside = returns[returns < rf_daily]
        _down_std = float(np.std(downside)) if len(downside) > 1 else 0.0
        sortino = (
            float(excess.mean() / _down_std * np.sqrt(TRADING_DAYS_PER_YEAR))
            if _down_std > EPSILON
            else 0.0
        )

        # ── Drawdown ──────────────────────────────────────────────────────────────
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(dd.min())
        avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

        # Max drawdown duration (consecutive days underwater)
        in_dd = dd < 0
        max_dur = 0
        cur_dur = 0
        for v in in_dd:
            cur_dur = cur_dur + 1 if v else 0
            max_dur = max(max_dur, cur_dur)

        # ── Calmar ────────────────────────────────────────────────────────────────
        years = len(df) / TRADING_DAYS_PER_YEAR
        ann_return = float(
            (equity[-1] / initial_equity) ** (1 / years) - 1
        ) if years > 0 else 0.0
        calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else np.inf

        # Additional metrics
        omega = _omega_ratio(returns, OMEGA_THRESHOLD)
        ulcer = _ulcer_index(equity)

        # Trading statistics
        num_trades = int(trade_mask.sum())
        wins = df.loc[df["pnl"] > 0, "pnl"]
        losses = df.loc[df["pnl"] < 0, "pnl"]
        win_rate = float(wins.count() / num_trades) if num_trades > 0 else 0.0
        avg_win_pct = float(wins.mean()) if not wins.empty else 0.0
        avg_loss_pct = float(losses.mean()) if not losses.empty else 0.0
        profit_factor = float(wins.sum() / -losses.sum()) if losses.sum() != 0 else np.inf
        expectancy = float(avg_win_pct * win_rate + avg_loss_pct * (1 - win_rate))

        # Equity curve for charting
        equity_curve = [
            {"date": str(idx.date()), "equity": val}
            for idx, val in zip(df.index, equity)
        ]

        return BacktestMetrics(
            total_return=equity[-1] / initial_equity - 1,
            annualized_return=ann_return,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            omega_ratio=omega,
            ulcer_index=ulcer,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration_days=max_dur,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            profit_factor=profit_factor,
            expectancy=expectancy,
            equity_curve=equity_curve,
        )
    except Exception as e:
        logger.error("Backtest execution failed: %s", e, exc_info=True)
        raise