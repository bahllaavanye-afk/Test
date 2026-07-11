from __future__ import annotations

import logging
import unittest
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
            (equity[-1] / equity[0]) ** (1 / years) - 1
            if years > 0
            else 0.0
        )
        calmar = float(ann_return / abs(max_dd) if max_dd != 0 else 0.0)

        # ── Omega Ratio & Ulcer Index ─────────────────────────────────────────────
        omega = _omega_ratio(returns)
        ulcer = _ulcer_index(equity)

        # ── Trading statistics ───────────────────────────────────────────────────
        trades_executed = trade_mask.sum()
        wins = df["pnl"][df["pnl"] > 0]
        losses = df["pnl"][df["pnl"] < 0]
        win_rate = float(wins.count() / trades_executed) if trades_executed > 0 else 0.0
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        profit_factor = float(
            (wins.sum() / -losses.sum()) if losses.sum() != 0 else np.inf
        )
        expectancy = float(avg_win * win_rate - avg_loss * (1 - win_rate))

        # Compile metrics
        metrics = BacktestMetrics(
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
            num_trades=int(trades_executed),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            equity_curve=df[["equity"]].reset_index().to_dict(orient="records"),
        )
        return metrics
    except Exception as e:
        logger.error("Backtest execution error: %s", e, exc_info=True)
        raise


class TestBacktestEngine(unittest.TestCase):
    """Edge‑case unit tests for the backtest engine."""

    def setUp(self) -> None:
        self.dates = pd.date_range(start="2022-01-03", periods=5, freq="B")
        self.prices = pd.Series([100, 101, 102, 101, 100], index=self.dates)
        self.signals = pd.Series([0, 1, 0, -1, 0], index=self.dates)
        self.opens = pd.Series([99, 100, 101, 100, 99], index=self.dates)
        self.volume = pd.Series([1_000_000, 1_200_000, 800_000, 900_000, 1_100_000], index=self.dates)

    def test_empty_signals_raises(self) -> None:
        empty_signals = pd.Series([], dtype=float)
        with self.assertRaises(ValueError) as cm:
            run_backtest(empty_signals, self.prices)
        self.assertIn("signals series is empty", str(cm.exception))

    def test_mismatched_index_raises(self) -> None:
        mismatched_signals = pd.Series([1, -1, 0], index=pd.date_range("2022-01-01", periods=3))
        with self.assertRaises(ValueError) as cm:
            run_backtest(mismatched_signals, self.prices)
        self.assertIn("signals and prices must share the same index", str(cm.exception))

    def test_adaptive_slippage_flat_when_volume_none(self) -> None:
        trade_size = pd.Series([10_000, 20_000], index=[0, 1])
        slip_series = _adaptive_slippage(trade_size, None, base_slippage_pct=0.001)
        self.assertTrue((slip_series == 0.001).all())
        self.assertEqual(slip_series.index.tolist(), [0, 1])

    def test_omega_ratio_infinite_when_no_losses(self) -> None:
        returns = np.array([0.02, 0.03, 0.01])  # all positive
        omega = _omega_ratio(returns, threshold=0.0)
        self.assertTrue(np.isinf(omega))

    def test_ulcer_index_zero_when_no_drawdown(self) -> None:
        equity = np.array([100_000, 100_000, 100_000])
        ulcer = _ulcer_index(equity)
        self.assertAlmostEqual(ulcer, 0.0)


if __name__ == "__main__":
    unittest.main()