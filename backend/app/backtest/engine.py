from __future__ import annotations

import logging
from dataclasses import dataclass, field
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


def _max_drawdown_duration(dd: np.ndarray) -> int:
    """
    Vectorized calculation of the longest consecutive period where drawdown is negative.
    """
    # Convert boolean mask to int (1 where in drawdown, 0 otherwise)
    mask = (dd < 0).astype(int)
    # Compute lengths of consecutive ones using cumulative sum resetting at zeros
    # Where mask == 0, reset cumulative count to 0
    cum = np.cumsum(mask)
    reset = np.where(mask == 0, cum, 0)
    # The length of each run is cum - previous reset point
    run_lengths = cum - np.maximum.accumulate(reset)
    return int(run_lengths.max()) if run_lengths.size else 0


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

        # Carry forward last non‑zero signal to maintain position
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        # Shift so position change takes effect at *next* bar's open
        df["position"] = df["position"].shift(1).fillna(0)

        # Detect transitions (direction changes or new entries)
        df["trade"] = df["position"].diff().fillna(0)
        trade_mask = df["trade"] != 0

        # Volume‑adaptive slippage on transition bars only
        trade_size_usd = df["trade"].abs() * df["fill_price"] * initial_equity / df["fill_price"].iloc[0]
        slip = _adaptive_slippage(trade_size_usd, _volume_usd, slippage_pct)

        total_cost_pct = (commission_pct + slip) * trade_mask.astype(float)

        # Daily P&L: mark‑to‑market returns on the held position
        df["bar_return"] = df["price"].pct_change().fillna(0)
        df["pnl"] = df["position"] * df["bar_return"] - total_cost_pct

        df["equity"] = initial_equity * (1 + df["pnl"]).cumprod()
        df["equity"] = df["equity"].ffill().fillna(initial_equity)

        equity = df["equity"].values
        returns = df["pnl"].values
        rf_daily = risk_free_annual / 252.0

        # ── Sharpe ────────────────────────────────────────────────────────────────
        excess = returns - rf_daily
        _excess_std = float(np.std(excess, ddof=1))
        sharpe = (
            float(excess.mean() / _excess_std * np.sqrt(252))
            if _excess_std > 1e-10
            else 0.0
        )

        # ── Sortino ───────────────────────────────────────────────────────────────
        downside = returns[returns < rf_daily]
        _down_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
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
        max_dd_duration = _max_drawdown_duration(dd)

        # ── Calmar ────────────────────────────────────────────────────────────────
        years = len(df) / 252.0 if len(df) >= 252 else len(df) / 252.0
        total_return = float(equity[-1] / initial_equity - 1)
        annualized_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else np.inf

        # ── Omega & Ulcer Index ───────────────────────────────────────────────────
        omega = _omega_ratio(returns)
        ulcer = _ulcer_index(equity)

        # ── Trade statistics ─────────────────────────────────────────────────────
        trades = df.loc[trade_mask, "trade"]
        num_trades = int(trades.abs().sum())
        if num_trades > 0:
            pnl_trades = df.loc[trade_mask, "pnl"]
            wins = pnl_trades[pnl_trades > 0]
            losses = pnl_trades[pnl_trades < 0]
            win_rate = float(wins.count() / num_trades)
            avg_win = float(wins.mean()) if not wins.empty else 0.0
            avg_loss = float(losses.mean()) if not losses.empty else 0.0
            profit_factor = float(wins.sum() / -losses.sum()) if losses.sum() != 0 else np.inf
            expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)
        else:
            win_rate = avg_win = avg_loss = profit_factor = expectancy = 0.0

        # ── Equity curve for charting ─────────────────────────────────────────────
        equity_curve = [
            {"date": str(idx.date()), "equity": val}
            for idx, val in zip(df.index, equity)
        ]

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            omega_ratio=omega,
            ulcer_index=ulcer,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration_days=max_dd_duration,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            equity_curve=equity_curve,
        )
    except Exception as e:
        logger.error("Backtest execution failed: %s", e, exc_info=True)
        raise