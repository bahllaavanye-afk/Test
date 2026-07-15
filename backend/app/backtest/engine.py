from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """
    Container for backtest performance metrics.

    Attributes
    ----------
    total_return : float
        Cumulative return of the strategy over the backtest period.
    annualized_return : float
        Annualized version of ``total_return`` assuming 252 trading days per year.
    sharpe : float
        Annualized Sharpe ratio (mean excess return / standard deviation of excess returns).
    sortino : float
        Annualized Sortino ratio (mean excess return / downside deviation).
    calmar : float
        Ratio of annualized return to maximum drawdown (absolute value).
    omega_ratio : float
        Probability‑weighted gain to loss ratio above a threshold (default threshold = 0).
    ulcer_index : float
        Root‑mean‑square of percentage drawdown, penalising prolonged underwater periods.
    max_drawdown : float
        Maximum drawdown observed during the backtest (negative value).
    avg_drawdown : float
        Average drawdown (only periods where equity is below its peak).
    max_drawdown_duration_days : int
        Longest consecutive stretch of days where equity was below its peak.
    num_trades : int
        Total number of executed trades (entries + exits).
    win_rate : float
        Proportion of trades that were profitable.
    avg_win_pct : float
        Average percentage profit of winning trades.
    avg_loss_pct : float
        Average percentage loss of losing trades (positive number).
    profit_factor : float
        Ratio of gross profit to gross loss.
    expectancy : float
        Expected return per trade: ``avg_win * win_rate - avg_loss * (1 - win_rate)``.
    equity_curve : List[Dict[str, Any]]
        Time‑series representation of equity for charting; each entry contains a ``date`` and ``equity``.
    """

    # Returns
    total_return: float
    annualized_return: float

    # Risk‑adjusted
    sharpe: float
    sortino: float
    calmar: float
    omega_ratio: float
    ulcer_index: float

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
    expectancy: float

    # Equity curve (for charting)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)


def _omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Compute the Omega ratio.

    The Omega ratio is defined as the sum of gains above ``threshold`` divided by the sum
    of losses below ``threshold``.  It provides a probability‑weighted view of upside versus
    downside.

    Parameters
    ----------
    returns : np.ndarray
        Array of period returns.
    threshold : float, optional
        Minimum acceptable return; defaults to 0.0.

    Returns
    -------
    float
        The Omega ratio; ``inf`` if there are no losses.
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    if losses.sum() == 0:
        return float("inf")
    return float(gains.sum() / losses.sum())


def _ulcer_index(equity: np.ndarray) -> float:
    """
    Compute the Ulcer Index.

    The Ulcer Index is the root‑mean‑square of percentage drawdown, placing greater
    emphasis on prolonged periods below the equity peak.

    Parameters
    ----------
    equity : np.ndarray
        Cumulative equity curve.

    Returns
    -------
    float
        The Ulcer Index value.
    """
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / peak * 100.0
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def _adaptive_slippage(
    trade_size_usd: pd.Series,
    volume_usd: Optional[pd.Series],
    base_slippage_pct: float,
) -> pd.Series:
    """
    Compute slippage that adapts to daily market volume using Kyle's square‑root model.

    The model scales slippage with the square root of the participation rate
    (trade size divided by daily dollar volume).  The result is capped at five times
    the base slippage to guard against extreme illiquid days.

    Parameters
    ----------
    trade_size_usd : pd.Series
        Dollar value of each trade (absolute, aligned with the backtest index).
    volume_usd : pd.Series or None
        Daily dollar volume; if ``None`` or all zeros a flat slippage is applied.
    base_slippage_pct : float
        Baseline slippage expressed as a fraction of price (e.g., 0.0005 for 5 bps).

    Returns
    -------
    pd.Series
        Slippage percentage for each bar, aligned with ``trade_size_usd``.
    """
    if volume_usd is None or (volume_usd == 0).all():
        return pd.Series(base_slippage_pct, index=trade_size_usd.index)

    participation = (trade_size_usd / volume_usd.clip(lower=1)).clip(0, 1)
    scaled = base_slippage_pct * np.sqrt(participation)
    return scaled.clip(upper=base_slippage_pct * 5)


def run_backtest(
    signals: pd.Series,
    prices: pd.Series,
    opens: Optional[pd.Series] = None,
    volume: Optional[pd.Series] = None,
    initial_equity: float = 100_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    fill_at_open: bool = True,
    risk_free_annual: float = 0.05,
) -> BacktestMetrics:
    """
    Execute a vectorised backtest.

    The function assumes that ``signals`` have already been shifted to avoid look‑ahead bias.
    It supports optional adaptive slippage based on daily volume and optional execution at
    the next bar's open price.

    Parameters
    ----------
    signals : pd.Series
        Trading signals (+1 for long, -1 for short, 0 for flat). Must be aligned with ``prices``.
    prices : pd.Series
        Close prices used for mark‑to‑market valuation.
    opens : pd.Series, optional
        Open prices; used when ``fill_at_open`` is ``True``. Falls back to ``prices`` if omitted.
    volume : pd.Series, optional
        Daily traded volume (shares or contracts). Enables adaptive slippage when provided.
    initial_equity : float, default 100_000.0
        Starting capital for the backtest.
    commission_pct : float, default 0.001
        Commission expressed as a fraction of trade notional (e.g., 0.001 = 10 bps).
    slippage_pct : float, default 0.0005
        Base slippage as a fraction of price; scaled adaptively when ``volume`` is supplied.
    fill_at_open : bool, default True
        If ``True``, position changes are executed at the next bar's open price.
    risk_free_annual : float, default 0.05
        Annual risk‑free rate used for Sharpe and Sortino calculations.

    Returns
    -------
    BacktestMetrics
        Aggregated performance metrics for the backtest period.
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
        volume_usd: Optional[pd.Series] = None
        if volume is not None:
            volume_usd = volume.reindex(df.index).fillna(0) * df["price"]

        # Carry forward last non‑zero signal to maintain position
        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        # Shift so position change takes effect at *next* bar's open
        df["position"] = df["position"].shift(1).fillna(0)

        # Detect transitions (direction changes or new entries)
        df["trade"] = df["position"].diff().fillna(0)
        trade_mask = df["trade"] != 0

        # Volume‑adaptive slippage on transition bars only
        trade_size_usd = df["trade"].abs() * df["fill_price"] * initial_equity / df["fill_price"].iloc[0]
        slip = _adaptive_slippage(trade_size_usd, volume_usd, slippage_pct)

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
        years = len(df) / 252.0 if len(df) > 0 else 0.0
        ann_return = float(equity[-1] / initial_equity - 1) / years if years > 0 else 0.0
        calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0

        # ── Omega Ratio & Ulcer Index ─────────────────────────────────────────────
        omega = _omega_ratio(returns)
        ulcer = _ulcer_index(equity)

        # ── Trade‑level statistics ───────────────────────────────────────────────
        trade_pnl = df.loc[trade_mask, "pnl"]
        num_trades = int(trade_pnl.shape[0])
        wins = trade_pnl[trade_pnl > 0]
        losses = trade_pnl[trade_pnl < 0]

        win_rate = float(wins.shape[0] / num_trades) if num_trades > 0 else 0.0
        avg_win_pct = float(wins.mean()) if wins.shape[0] > 0 else 0.0
        avg_loss_pct = float(-losses.mean()) if losses.shape[0] > 0 else 0.0

        gross_profit = float(wins.sum()) if wins.shape[0] > 0 else 0.0
        gross_loss = float(-losses.sum()) if losses.shape[0] > 0 else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        expectancy = float(avg_win_pct * win_rate - avg_loss_pct * (1 - win_rate))

        # ── Equity curve for charting ─────────────────────────────────────────────
        equity_curve: List[Dict[str, Any]] = [
            {"date": idx, "equity": val} for idx, val in zip(df.index, equity)
        ]

        total_return = float(equity[-1] / initial_equity - 1)
        annualized_return = ann_return

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
    except Exception as e:
        logger.error("Unexpected error during backtest execution: %s", e, exc_info=True)
        raise