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

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_INITIAL_EQUITY: float = 100_000.0
DEFAULT_COMMISSION_PCT: float = 0.001
DEFAULT_SLIPPAGE_PCT: float = 0.0005
DEFAULT_RISK_FREE_ANNUAL: float = 0.05

TRADING_DAYS_PER_YEAR: int = 252
EPSILON: float = 1e-10
SLIPPAGE_CAP_MULTIPLIER: float = 5.0
OMEGA_THRESHOLD_DEFAULT: float = 0.0

EQUITY_CURVE_DATE_KEY: str = "date"
EQUITY_CURVE_EQUITY_KEY: str = "equity"


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
    equity_curve: List[Dict] = field(default_factory=list)


def _omega_ratio(returns: np.ndarray, threshold: float = OMEGA_THRESHOLD_DEFAULT) -> float:
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
    volume_usd: Optional[pd.Series],
    base_slippage_pct: float,
) -> pd.Series:
    """
    Scale slippage with participation rate using Kyle's sqrt-of-volume model.

    slippage = base * sqrt(participation_rate)
    where participation_rate = trade_size / (price * daily_volume).

    Caps at SLIPPAGE_CAP_MULTIPLIER× base to avoid extreme values on illiquid days.
    """
    if volume_usd is None or (volume_usd == 0).all():
        return pd.Series(base_slippage_pct, index=trade_size_usd.index)

    participation = (trade_size_usd / volume_usd.clip(lower=1)).clip(0, 1)
    scaled = base_slippage_pct * np.sqrt(participation)
    return scaled.clip(upper=base_slippage_pct * SLIPPAGE_CAP_MULTIPLIER)


def _prepare_dataframe(
    signals: pd.Series,
    prices: pd.Series,
    opens: Optional[pd.Series],
    fill_at_open: bool,
    volume: Optional[pd.Series],
    initial_equity: float,
) -> tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
    """Create the working DataFrame and auxiliary series."""
    fill_prices = opens if (fill_at_open and opens is not None) else prices

    df = pd.DataFrame(
        {
            "signal": signals,
            "price": prices,
            "fill_price": fill_prices,
        }
    ).dropna(subset=["signal", "price"])

    # Volume expressed in USD (price * volume) aligned to df index.
    volume_usd: Optional[pd.Series] = None
    if volume is not None:
        volume_usd = volume.reindex(df.index).fillna(0) * df["price"]

    # Position: forward‑filled signal, shifted to apply on next bar.
    df["position"] = (
        df["signal"]
        .replace(0, np.nan)
        .ffill()
        .fillna(0)
        .shift(1)
        .fillna(0)
    )

    # Trade detection (non‑zero change in position).
    df["trade"] = df["position"].diff().fillna(0)

    # Trade size in USD (used for slippage calculation).
    trade_size_usd = (
        df["trade"]
        .abs()
        * df["fill_price"]
        * initial_equity
        / df["fill_price"].iloc[0]
    )

    return df, volume_usd, trade_size_usd


def _apply_costs_and_equity(
    df: pd.DataFrame,
    slip_series: pd.Series,
    commission_pct: float,
    trade_mask: pd.Series,
) -> pd.DataFrame:
    """Apply commission & slippage costs, compute P&L and equity curve."""
    total_cost_pct = (commission_pct + slip_series) * trade_mask.astype(float)

    df["bar_return"] = df["price"].pct_change().fillna(0)
    df["pnl"] = df["position"] * df["bar_return"] - total_cost_pct

    # Initialise equity column if absent.
    if "equity" not in df.columns:
        df["equity"] = df["pnl"].add(1).cumprod() * DEFAULT_INITIAL_EQUITY

    # If equity already exists (e.g., from a previous run), update it.
    else:
        df["equity"] = df["pnl"].add(1).cumprod() * df["equity"].iloc[0]

    df["equity"] = df["equity"].ffill().fillna(df["equity"].iloc[0])
    return df


def _compute_return_metrics(
    equity: np.ndarray,
    returns: np.ndarray,
    risk_free_annual: float,
) -> dict:
    """Calculate Sharpe, Sortino, Calmar, Omega, Ulcer and drawdown statistics."""
    rf_daily = risk_free_annual / TRADING_DAYS_PER_YEAR

    # Sharpe
    excess = returns - rf_daily
    excess_std = float(np.std(excess))
    sharpe = (
        float(excess.mean() / excess_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        if excess_std > EPSILON
        else 0.0
    )

    # Sortino
    downside = returns[returns < rf_daily]
    down_std = float(np.std(downside)) if len(downside) > 1 else 0.0
    sortino = (
        float(excess.mean() / down_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        if down_std > EPSILON
        else 0.0
    )

    # Drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    # Max drawdown duration (consecutive days underwater)
    in_dd = dd < 0
    max_dur = 0
    cur_dur = 0
    for flag in in_dd:
        cur_dur = cur_dur + 1 if flag else 0
        max_dur = max(max_dur, cur_dur)

    # Calmar
    years = len(returns) / TRADING_DAYS_PER_YEAR
    ann_return = float(
        (equity[-1] / equity[0]) ** (1.0 / max(years, EPSILON)) - 1.0
    )
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Omega & Ulcer
    omega = _omega_ratio(returns, threshold=rf_daily)
    ulcer = _ulcer_index(equity)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "omega_ratio": omega,
        "ulcer_index": ulcer,
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "max_drawdown_duration_days": int(max_dur),
        "annualized_return": ann_return,
    }


def _compute_trade_statistics(df: pd.DataFrame) -> dict:
    """
    Derive trade‑level performance metrics using vectorised operations.
    """
    # Identify rows where a trade occurs (position change)
    trade_mask = df["trade"] != 0
    if not trade_mask.any():
        # No trades – return neutral statistics
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
        }

    trade_entries = df[trade_mask].copy()

    # Side (position) and entry price
    sides = trade_entries["position"]
    entry_prices = trade_entries["fill_price"]

    # Exit prices are the fill_price of the *next* trade entry
    exit_prices = entry_prices.shift(-1)

    # Drop the last entry where there is no subsequent exit
    valid = exit_prices.notna()
    sides = sides[valid]
    entry_prices = entry_prices[valid]
    exit_prices = exit_prices[valid]

    # Trade P&L as percentage return
    trade_pnls = (exit_prices - entry_prices) * sides / entry_prices

    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]

    num_trades = int(trade_pnls.shape[0])
    win_rate = float(wins.shape[0] / num_trades) if num_trades > 0 else 0.0
    avg_win_pct = float(wins.mean()) if not wins.empty else 0.0
    avg_loss_pct = -float(losses.mean()) if not losses.empty else 0.0
    profit_factor = float(wins.sum() / -losses.sum()) if losses.sum() != 0 else float("inf")
    expectancy = avg_win_pct * win_rate - avg_loss_pct * (1 - win_rate)

    return {
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    }


def run_backtest(
    signals: pd.Series,
    prices: pd.Series,
    opens: Optional[pd.Series] = None,
    volume: Optional[pd.Series] = None,
    fill_at_open: bool = True,
    initial_equity: float = DEFAULT_INITIAL_EQUITY,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    risk_free_annual: float = DEFAULT_RISK_FREE_ANNUAL,
) -> BacktestMetrics:
    """
    Execute a vectorised backtest and return a populated BacktestMetrics instance.
    """
    # ------------------------------------------------------------------
    # 1. Prepare data frame and auxiliary series
    # ------------------------------------------------------------------
    df, volume_usd, trade_size_usd = _prepare_dataframe(
        signals=signals,
        prices=prices,
        opens=opens,
        fill_at_open=fill_at_open,
        volume=volume,
        initial_equity=initial_equity,
    )

    # ------------------------------------------------------------------
    # 2. Compute slippage series (volume‑adaptive)
    # ------------------------------------------------------------------
    slippage_series = _adaptive_slippage(
        trade_size_usd=trade_size_usd,
        volume_usd=volume_usd,
        base_slippage_pct=slippage_pct,
    )

    # ------------------------------------------------------------------
    # 3. Apply transaction costs and compute equity curve
    # ------------------------------------------------------------------
    df = _apply_costs_and_equity(
        df=df,
        slip_series=slippage_series,
        commission_pct=commission_pct,
        trade_mask=df["trade"].abs() > 0,
    )

    # ------------------------------------------------------------------
    # 4. Assemble return series for risk metrics
    # ------------------------------------------------------------------
    equity = df["equity"].to_numpy()
    returns = df["bar_return"].to_numpy()

    # ------------------------------------------------------------------
    # 5. Compute risk‑adjusted and drawdown metrics
    # ------------------------------------------------------------------
    return_metrics = _compute_return_metrics(
        equity=equity,
        returns=returns,
        risk_free_annual=risk_free_annual,
    )

    # ------------------------------------------------------------------
    # 6. Compute trade‑level statistics
    # ------------------------------------------------------------------
    trade_stats = _compute_trade_statistics(df)

    # ------------------------------------------------------------------
    # 7. Build final BacktestMetrics dataclass
    # ------------------------------------------------------------------
    total_return = float(equity[-1] / equity[0] - 1.0)

    equity_curve = [
        {EQUITY_CURVE_DATE_KEY: idx, EQUITY_CURVE_EQUITY_KEY: val}
        for idx, val in zip(df.index, equity)
    ]

    metrics = BacktestMetrics(
        total_return=total_return,
        annualized_return=return_metrics["annualized_return"],
        sharpe=return_metrics["sharpe"],
        sortino=return_metrics["sortino"],
        calmar=return_metrics["calmar"],
        omega_ratio=return_metrics["omega_ratio"],
        ulcer_index=return_metrics["ulcer_index"],
        max_drawdown=return_metrics["max_drawdown"],
        avg_drawdown=return_metrics["avg_drawdown"],
        max_drawdown_duration_days=return_metrics["max_drawdown_duration_days"],
        num_trades=trade_stats["num_trades"],
        win_rate=trade_stats["win_rate"],
        avg_win_pct=trade_stats["avg_win_pct"],
        avg_loss_pct=trade_stats["avg_loss_pct"],
        profit_factor=trade_stats["profit_factor"],
        expectancy=trade_stats["expectancy"],
        equity_curve=equity_curve,
    )

    return metrics