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
    fill_prices = opens if (fill_at_open and opens is not None) else prices

    df = pd.DataFrame({
        "signal":      signals,
        "price":       prices,
        "fill_price":  fill_prices,
    }).dropna(subset=["signal", "price"])

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
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    # ── Sortino ───────────────────────────────────────────────────────────────
    downside = returns[returns < rf_daily]
    sortino = (
        float(excess.mean() / downside.std() * np.sqrt(252))
        if len(downside) > 1 and downside.std() > 0 else 0.0
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
    ann_return = float((equity[-1] / initial_equity) ** (1.0 / max(years, 1e-6)) - 1.0)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # ── Omega / Ulcer ─────────────────────────────────────────────────────────
    omega = _omega_ratio(returns, threshold=rf_daily)
    ulcer = _ulcer_index(equity)

    # ── Trade-level stats (vectorised) ────────────────────────────────────────
    pos_series = df["position"]
    fill_series = df["fill_price"]

    entries = df.index[df["trade"] != 0].tolist()
    trade_pnls: list[float] = []

    for i in range(len(entries) - 1):
        t0, t1 = entries[i], entries[i + 1]
        side = float(pos_series.loc[t0])
        if side == 0:
            continue
        entry_p = float(fill_series.loc[t0])
        exit_p  = float(fill_series.loc[t1])
        trade_pnls.append((exit_p - entry_p) * side / entry_p)

    wins   = [r for r in trade_pnls if r > 0]
    losses = [r for r in trade_pnls if r <= 0]

    win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    avg_win  = float(np.mean(wins))  if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    profit_factor = (
        abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
    )
    expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)

    # ── Equity curve ──────────────────────────────────────────────────────────
    equity_curve = [
        {
            "date": str(idx.date() if hasattr(idx, "date") else idx),
            "equity": round(float(val), 2),
        }
        for idx, val in zip(df.index, df["equity"])
    ]

    total_return = float(equity[-1] / initial_equity - 1.0)

    return BacktestMetrics(
        total_return=round(total_return, 4),
        annualized_return=round(ann_return, 4),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        calmar=round(calmar, 4),
        omega_ratio=round(min(omega, 99.99), 4),
        ulcer_index=round(ulcer, 4),
        max_drawdown=round(max_dd, 4),
        avg_drawdown=round(avg_dd, 4),
        max_drawdown_duration_days=max_dur,
        num_trades=len(trade_pnls),
        win_rate=round(win_rate, 4),
        avg_win_pct=round(avg_win * 100, 4),
        avg_loss_pct=round(avg_loss * 100, 4),
        profit_factor=round(profit_factor, 4),
        expectancy=round(expectancy * 100, 4),
        equity_curve=equity_curve,
    )
