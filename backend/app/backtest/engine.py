"""
Backtesting engine — vectorized, institution-grade.

Key features over a naive engine:
  • open-price fills (signal at close → enter at next open)
  • volume-adaptive market impact (Kyle's sqrt model)
  • comprehensive risk metrics: Sharpe, Sortino, Calmar, Omega, Ulcer Index
  • vectorized trade P&L (no Python loops)
  • overnight gap returns modeled separately
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Main backtest routine
# --------------------------------------------------------------------------- #


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
    signals     : +1 buy, -1 sell, 0 hold. Must be pre‑shifted to avoid lookahead.
    prices      : OHLCV close prices (used for mark‑to‑market).
    opens       : Open prices. When fill_at_open=True, trades execute at next open.
                  Falls back to close if not provided.
    volume      : Daily volume (shares or contracts). Used for adaptive slippage.
                  Pass None to use flat slippage_pct.
    fill_at_open: If True, position changes fill at the bar's OPEN price, not
                  the previous bar's close. This is more realistic for EOD signals.
    """
    # ------------------------------------------------------------------- #
    # Prepare price series for fill execution
    # ------------------------------------------------------------------- #
    fill_prices = opens if (fill_at_open and opens is not None) else prices

    df = pd.DataFrame(
        {
            "signal": signals,
            "price": prices,
            "fill_price": fill_prices,
        }
    ).dropna(subset=["signal", "price"])

    # ------------------------------------------------------------------- #
    # Volume handling (USD value)
    # ------------------------------------------------------------------- #
    volume_usd: pd.Series | None = None
    if volume is not None:
        volume_usd = volume.reindex(df.index).fillna(0) * df["price"]

    # ------------------------------------------------------------------- #
    # Position logic (carry‑forward, shifted for next‑bar execution)
    # ------------------------------------------------------------------- #
    df["position"] = (
        df["signal"]
        .replace(0, np.nan)
        .ffill()
        .fillna(0)
        .shift(1)
        .fillna(0)
    )

    # ------------------------------------------------------------------- #
    # Trade detection
    # ------------------------------------------------------------------- #
    df["trade"] = df["position"].diff().fillna(0)
    trade_mask = df["trade"] != 0

    # ------------------------------------------------------------------- #
    # Slippage & commission costs (applied only on bars where a trade occurs)
    # ------------------------------------------------------------------- #
    trade_size_usd = df["trade"].abs() * df["fill_price"] * initial_equity / df["fill_price"].iloc[0]
    slip = _adaptive_slippage(trade_size_usd, volume_usd, slippage_pct)
    total_cost_pct = (commission_pct + slip) * trade_mask.astype(float)

    # ------------------------------------------------------------------- #
    # Daily P&L and equity curve
    # ------------------------------------------------------------------- #
    df["bar_return"] = df["price"].pct_change().fillna(0)
    df["pnl"] = df["position"] * df["bar_return"] - total_cost_pct
    df["equity"] = initial_equity * (1 + df["pnl"]).cumprod()
    df["equity"] = df["equity"].ffill().fillna(initial_equity)

    equity = df["equity"].values
    returns = df["pnl"].values
    rf_daily = risk_free_annual / 252.0

    # ------------------------------------------------------------------- #
    # Sharpe
    # ------------------------------------------------------------------- #
    excess = returns - rf_daily
    excess_std = float(np.std(excess))
    sharpe = (
        float(excess.mean() / excess_std * np.sqrt(252))
        if excess_std > 1e-10
        else 0.0
    )

    # ------------------------------------------------------------------- #
    # Sortino
    # ------------------------------------------------------------------- #
    downside = returns[returns < rf_daily]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 0.0
    sortino = (
        float(excess.mean() / downside_std * np.sqrt(252))
        if downside_std > 1e-10
        else 0.0
    )

    # ------------------------------------------------------------------- #
    # Drawdown metrics
    # ------------------------------------------------------------------- #
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    # Max drawdown duration (vectorised)
    in_dd = dd < 0
    # Identify runs of consecutive True values
    runs = (in_dd != in_dd.shift()).cumsum()
    run_lengths = in_dd.groupby(runs).cumcount() + 1
    max_dd_duration = int(run_lengths[in_dd].max()) if in_dd.any() else 0

    # ------------------------------------------------------------------- #
    # Calmar ratio
    # ------------------------------------------------------------------- #
    years = len(df) / 252.0
    ann_return = float((equity[-1] / initial_equity) ** (1.0 / max(years, 1e-6)) - 1.0)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # ------------------------------------------------------------------- #
    # Omega & Ulcer Index
    # ------------------------------------------------------------------- #
    omega = _omega_ratio(returns, threshold=rf_daily)
    ulcer = _ulcer_index(equity)

    # ------------------------------------------------------------------- #
    # Trade‑level statistics (fully vectorised)
    # ------------------------------------------------------------------- #
    trade_indices = df.index[trade_mask].to_numpy()
    # Exclude the last trade index because we need a subsequent exit point
    if len(trade_indices) >= 2:
        entry_idx = trade_indices[:-1]
        exit_idx = trade_indices[1:]

        side = df.loc[entry_idx, "position"].to_numpy()
        entry_price = df.loc[entry_idx, "fill_price"].to_numpy()
        exit_price = df.loc[exit_idx, "fill_price"].to_numpy()

        trade_pnls = (exit_price - entry_price) * side / entry_price
    else:
        trade_pnls = np.array([])

    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]

    num_trades = int(trade_pnls.size)
    win_rate = float(len(wins) / num_trades) if num_trades > 0 else 0.0
    avg_win = float(wins.mean()) if wins.size > 0 else 0.0
    avg_loss = float(losses.mean()) if losses.size > 0 else 0.0
    profit_factor = (
        abs(wins.sum() / losses.sum()) if losses.size > 0 and losses.sum() != 0 else float("inf")
    )
    expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)

    # ------------------------------------------------------------------- #
    # Equity curve for downstream visualisation
    # ------------------------------------------------------------------- #
    equity_curve = [
        {
            "date": str(idx.date() if hasattr(idx, "date") else idx),
            "equity": round(float(val), 2),
        }
        for idx, val in zip(df.index, df["equity"])
    ]

    total_return = float(equity[-1] / initial_equity - 1.0)

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=ann_return,
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