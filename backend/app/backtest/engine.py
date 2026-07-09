from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestError(Exception):
    """Base exception for backtest failures."""


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
        if not isinstance(signals, pd.Series):
            raise TypeError("signals must be a pandas Series")
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        if signals.empty:
            raise ValueError("signals series is empty")
        if prices.empty:
            raise ValueError("prices series is empty")
        if not signals.index.equals(prices.index):
            raise ValueError("signals and prices must share the same index")

        if opens is not None:
            if not isinstance(opens, pd.Series):
                raise TypeError("opens must be a pandas Series when provided")
            if not opens.index.equals(prices.index):
                raise ValueError("opens must share the same index as prices")
        if volume is not None:
            if not isinstance(volume, pd.Series):
                raise TypeError("volume must be a pandas Series when provided")
            if not volume.index.equals(prices.index):
                raise ValueError("volume must share the same index as prices")
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
        rf_daily = risk_free_annual / 252.0

        # ── Sharpe ────────────────────────────────────────────────────────────────
        try:
            excess = returns - rf_daily
            _excess_std = float(np.std(excess))
            sharpe = (
                float(excess.mean() / _excess_std * np.sqrt(252))
                if _excess_std > 1e-10
                else 0.0
            )
        except Exception as e:
            logger.error("Sharpe calculation error: %s", e, exc_info=True)
            raise BacktestError("Failed to compute Sharpe ratio") from e

        # ── Sortino ───────────────────────────────────────────────────────────────
        try:
            downside = returns[returns < rf_daily]
            _down_std = float(np.std(downside)) if len(downside) > 1 else 0.0
            sortino = (
                float(excess.mean() / _down_std * np.sqrt(252))
                if _down_std > 1e-10
                else 0.0
            )
        except Exception as e:
            logger.error("Sortino calculation error: %s", e, exc_info=True)
            raise BacktestError("Failed to compute Sortino ratio") from e

        # ── Drawdown ──────────────────────────────────────────────────────────────
        try:
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
        except Exception as e:
            logger.error("Drawdown calculation error: %s", e, exc_info=True)
            raise BacktestError("Failed to compute drawdown metrics") from e

        # ── Calmar ────────────────────────────────────────────────────────────────
        try:
            years = len(df) / 252.0 if len(df) > 0 else 1.0
            ann_return = float(
                (equity[-1] / initial_equity) ** (1 / years) - 1
                if years > 0 else 0.0
            )
            calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else float("inf")
        except Exception as e:
            logger.error("Calmar calculation error: %s", e, exc_info=True)
            raise BacktestError("Failed to compute Calmar ratio") from e

        # ── Omega Ratio ────────────────────────────────────────────────────────
        try:
            omega = _omega_ratio(returns)
        except Exception as e:
            logger.error("Omega ratio calculation error: %s", e, exc_info=True)
            raise BacktestError("Failed to compute Omega ratio") from e

        # ── Ulcer Index ────────────────────────────────────────────────────────
        try:
            ulcer = _ulcer_index(equity)
        except Exception as e:
            logger.error("Ulcer index calculation error: %s", e, exc_info=True)
            raise BacktestError("Failed to compute Ulcer Index") from e

        # ── Trading statistics ─────────────────────────────────────────────────
        try:
            trade_returns = df.loc[trade_mask, "pnl"]
            num_trades = int(trade_mask.sum())
            win_trades = trade_returns[trade_returns > 0]
            loss_trades = trade_returns[trade_returns < 0]

            win_rate = float(len(win_trades) / num_trades) if num_trades > 0 else 0.0
            avg_win_pct = float(win_trades.mean()) if not win_trades.empty else 0.0
            avg_loss_pct = float(loss_trades.mean()) if not loss_trades.empty else 0.0
            profit_factor = (
                float(abs(win_trades.sum()) / abs(loss_trades.sum()))
                if not loss_trades.empty and loss_trades.sum() != 0
                else float("inf")
            )
            expectancy = float(
                avg_win_pct * win_rate + avg_loss_pct * (1 - win_rate)
            )
        except Exception as e:
            logger.error("Trading statistics calculation error: %s", e, exc_info=True)
            raise BacktestError("Failed to compute trading statistics") from e

        # ── Equity curve for charting ───────────────────────────────────────────
        try:
            equity_curve = [
                {"date": idx, "equity": val}
                for idx, val in zip(df.index, equity)
            ]
        except Exception as e:
            logger.error("Equity curve construction error: %s", e, exc_info=True)
            raise BacktestError("Failed to build equity curve") from e

        # ── Total & annualized return ───────────────────────────────────────────
        try:
            total_return = float(equity[-1] / initial_equity - 1)
            annualized_return = float(
                (equity[-1] / initial_equity) ** (1 / years) - 1
                if years > 0 else 0.0
            )
        except Exception as e:
            logger.error("Return calculation error: %s", e, exc_info=True)
            raise BacktestError("Failed to compute returns") from e

        metrics = BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
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
        return metrics

    except BacktestError:
        # Already logged; propagate upwards.
        raise
    except Exception as e:
        logger.exception("Unexpected error during backtest execution")
        raise BacktestError("Unexpected failure in backtest engine") from e