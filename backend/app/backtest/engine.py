from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def _omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
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

    Caps at 5× base to avoid extreme values on illiquid days.
    """
    if volume_usd is None or (volume_usd == 0).all():
        return pd.Series(base_slippage_pct, index=trade_size_usd.index)

    participation = (trade_size_usd / volume_usd.clip(lower=1)).clip(0, 1)
    scaled = base_slippage_pct * np.sqrt(participation)
    return scaled.clip(upper=base_slippage_pct * 5)


def run_backtest(
    signals: pd.Series,
    prices: pd.Series,
    opens: pd.Series | None = None,
    volume: pd.Series | None = None,
    initial_equity: float = 100_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    fill_at_open: bool = True,
    risk_free_annual: float = 0.05,
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
        # Series type checks
        if not isinstance(signals, pd.Series):
            raise ValueError("signals must be a pandas Series")
        if not isinstance(prices, pd.Series):
            raise ValueError("prices must be a pandas Series")
        if signals.empty:
            raise ValueError("signals series is empty")
        if prices.empty:
            raise ValueError("prices series is empty")
        if not signals.index.equals(prices.index):
            raise ValueError("signals and prices must share the same index")

        # Optional series checks
        if opens is not None:
            if not isinstance(opens, pd.Series):
                raise ValueError("opens must be a pandas Series when provided")
            if not opens.index.equals(prices.index):
                raise ValueError("opens must share the same index as prices")
        if volume is not None:
            if not isinstance(volume, pd.Series):
                raise ValueError("volume must be a pandas Series when provided")
            if not volume.index.equals(prices.index):
                raise ValueError("volume must share the same index as prices")

        # Numeric parameter checks
        if not isinstance(initial_equity, (int, float)):
            raise ValueError("initial_equity must be a numeric type")
        if initial_equity <= 0:
            raise ValueError("initial_equity must be greater than zero")
        if not isinstance(commission_pct, (int, float)):
            raise ValueError("commission_pct must be a numeric type")
        if commission_pct < 0:
            raise ValueError("commission_pct cannot be negative")
        if not isinstance(slippage_pct, (int, float)):
            raise ValueError("slippage_pct must be a numeric type")
        if slippage_pct < 0:
            raise ValueError("slippage_pct cannot be negative")
        if not isinstance(risk_free_annual, (int, float)):
            raise ValueError("risk_free_annual must be a numeric type")
        if risk_free_annual < 0:
            raise ValueError("risk_free_annual cannot be negative")
        if not isinstance(fill_at_open, bool):
            raise ValueError("fill_at_open must be a boolean")

        # Signal value validation
        allowed_signals = {-1, 0, 1}
        invalid_signals = set(signals.dropna().unique()) - allowed_signals
        if invalid_signals:
            raise ValueError(
                f"signals contain invalid values {invalid_signals}; allowed values are -1, 0, 1"
            )
    except ValueError as e:
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
        rf_daily = risk_free_annual / 252.0

        # ── Sharpe ────────────────────────────────────────────────────────────────
        excess = returns - rf_daily
        _excess_std = float(np.std(excess))
        sharpe = (
            float(excess.mean() / _excess_std * np.sqrt(252))
            if _excess_std > 1e-10
            else 0.0
        )

        # ── Sortino ───────────────────────────────────────────────────────────────
        downside = returns[returns < rf_daily]
        _down_std = float(np.std(downside)) if len(downside) > 1 else 0.0
        sortino = (
            float(excess.mean() / _down_std * np.sqrt(252))
            if _down_std > 1e-10
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
        years = len(df) / 252.0
        ann_return = float(
            (equity[-1] / initial_equity) ** (1 / years) - 1
            if years > 0
            else 0.0
        )
        calmar = float(ann_return / -max_dd) if max_dd != 0 else float("inf")

        # ── Omega Ratio & Ulcer Index ─────────────────────────────────────────────
        omega = _omega_ratio(returns)
        ulcer = _ulcer_index(equity)

        # ── Trading stats ───────────────────────────────────────────────────────
        trades = df["trade"].abs().sum()
        wins = df.loc[df["pnl"] > 0, "pnl"]
        losses = df.loc[df["pnl"] < 0, "pnl"]
        win_rate = float(wins.count() / trades) if trades > 0 else 0.0
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        profit_factor = float(-avg_win / avg_loss) if avg_loss != 0 else float("inf")
        expectancy = float(avg_win * win_rate + avg_loss * (1 - win_rate))

        # ── Assemble metrics ─────────────────────────────────────────────────────
        metrics = BacktestMetrics(
            total_return=float(equity[-1] / initial_equity - 1),
            annualized_return=ann_return,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            omega_ratio=omega,
            ulcer_index=ulcer,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration_days=int(max_dur),
            num_trades=int(trades),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            equity_curve=[
                {"date": str(idx.date()) if isinstance(idx, (pd.Timestamp, date)) else str(idx), "equity": eq}
                for idx, eq in zip(df.index, equity)
            ],
        )
        return metrics
    except Exception as e:
        logger.error("Unexpected error during backtest execution: %s", e, exc_info=True)
        raise