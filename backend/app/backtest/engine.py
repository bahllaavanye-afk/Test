"""
Backtesting engine using vectorized operations.
Returns equity curve, Sharpe, Sortino, max drawdown, win rate, profit factor.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date


@dataclass
class BacktestMetrics:
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    total_return: float
    annualized_return: float
    win_rate: float
    profit_factor: float
    num_trades: int
    equity_curve: list[dict]   # [{date, equity}, ...]


def run_backtest(
    signals: pd.Series,          # 1 buy, -1 sell, 0 hold — index is datetime
    prices: pd.Series,           # close prices aligned with signals
    initial_equity: float = 100_000,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> BacktestMetrics:
    """Vectorized backtest — signals must already be shifted to avoid lookahead."""
    df = pd.DataFrame({"signal": signals, "price": prices}).dropna()
    df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
    df["position"] = df["position"].shift(1).fillna(0)   # enter next bar

    df["trade"] = df["position"].diff().fillna(0)
    cost = df["trade"].abs() * df["price"] * (commission_pct + slippage_pct)
    df["pct_return"] = df["position"] * df["price"].pct_change().fillna(0) - cost / (initial_equity)

    df["equity"] = initial_equity * (1 + df["pct_return"]).cumprod()
    df["equity"] = df["equity"].ffill().fillna(initial_equity)

    returns = df["pct_return"].dropna()
    equity = df["equity"]

    # Drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    # Sharpe
    rf_daily = 0.05 / 252
    excess = returns - rf_daily
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    # Sortino
    downside = returns[returns < rf_daily]
    sortino = (excess.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0.0

    # Calmar
    years = len(df) / 252
    ann_return = (equity.iloc[-1] / initial_equity) ** (1 / max(years, 0.1)) - 1
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Trade-level stats
    trades = df[df["trade"] != 0].copy()
    trade_returns = []
    entry_price = None
    entry_side = 0
    for _, row in df.iterrows():
        if row["trade"] != 0 and entry_price is not None:
            pnl = (row["price"] - entry_price) * entry_side
            trade_returns.append(pnl)
        if row["trade"] != 0:
            entry_price = row["price"]
            entry_side = row["position"]

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

    equity_curve = [
        {"date": str(idx.date() if hasattr(idx, "date") else idx), "equity": round(val, 2)}
        for idx, val in equity.items()
    ]

    return BacktestMetrics(
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        calmar=round(calmar, 4),
        max_drawdown=round(max_dd, 4),
        total_return=round((equity.iloc[-1] / initial_equity - 1), 4),
        annualized_return=round(ann_return, 4),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        num_trades=len(trade_returns),
        equity_curve=equity_curve,
    )
