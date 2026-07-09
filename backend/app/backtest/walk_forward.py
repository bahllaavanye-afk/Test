"""Walk-forward validation: train on N years, test on M months, roll forward."""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from app.backtest.engine import run_backtest, BacktestMetrics

TIMEFRAME_TRAIN = 2  # years of training data
TIMEFRAME_TEST = 6  # months of testing data

MAX_EQUIITY = 100_000

@dataclass
class WalkForwardResult:
    windows: list[dict] = field(default_factory=list)
    avg_sharpe: float = 0.0
    avg_drawdown: float = 0.0
    combined_equity: list[dict] = field(default_factory=list)


def walk_forward(
    signals_fn,               # callable(train_df, test_df) -> pd.Series of signals on test_df
    prices: pd.Series,
    train_years: int | None = None,
    test_months: int | None = None,
) -> WalkForwardResult:
    """
    Rolls a train/test window across entire history.
    signals_fn receives (train_prices, test_prices) and must return signals for test period only.
    """
    train_bars = (train_years if train_years is not None else TIMEFRAME_TRAIN) * 252
    test_bars = (test_months if test_months is not None else TIMEFRAME_TEST) * 21
    result = WalkForwardResult()
    equity_carry = MAX_EQUIITY

    i = train_bars
    while i + test_bars <= len(prices):
        train = prices.iloc[i - train_bars:i]
        test = prices.iloc[i:i + test_bars]

        try:
            test_signals = signals_fn(train, test)
            metrics = run_backtest(test_signals, test, initial_equity=equity_carry)
            equity_carry = metrics.equity_curve[-1]["equity"] if metrics.equity_curve else equity_carry

            result.windows.append({
                "start": str(test.index[0].date()),
                "end": str(test.index[-1].date()),
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
                "num_trades": metrics.num_trades,
            })
            result.combined_equity.extend(metrics.equity_curve)
        except Exception as e:
            result.windows.append({"start": str(test.index[0].date()), "end": str(test.index[-1].date()), "error": str(e)})

        i += test_bars

    sharpes = [w["sharpe"] for w in result.windows if "sharpe" in w]
    dds = [w["max_drawdown"] for w in result.windows if "max_drawdown" in w]
    result.avg_sharpe = round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0
    result.avg_drawdown = round(sum(dds) / len(dds), 4) if dds else 0.0
    return result


# -------------------- Unit Tests --------------------
import unittest
from unittest.mock import patch
import datetime

class TestWalkForward(unittest.TestCase):
    def setUp(self):
        # Create a simple date index for testing
        self.dates = pd.date_range(start="2020-01-01", periods=500, freq="B")  # business days
        self.prices = pd.Series(range(500), index=self.dates)

    def dummy_signals(self, train, test):
        # Return a series of zeros matching the test length
        return pd.Series(0, index=test.index)

    @patch('backend.app.backtest.walk_forward.run_backtest')
    def test_single_window_boundary(self, mock_run):
        """When data length equals train + test exactly, a single window should be processed."""
        # Prepare mock metric
        mock_metric = unittest.mock.Mock()
        mock_metric.equity_curve = [{"equity": MAX_EQUIITY}]
        mock_metric.sharpe = 1.5
        mock_metric.max_drawdown = 0.1
        mock_metric.total_return = 0.05
        mock_metric.num_trades = 2
        mock_run.return_value = mock_metric

        train_bars = TIMEFRAME_TRAIN * 252
        test_bars = TIMEFRAME_TEST * 21
        # Trim prices to exact length needed
        prices = self.prices.iloc[:train_bars + test_bars]

        result = walk_forward(self.dummy_signals, prices)

        self.assertEqual(len(result.windows), 1)
        self.assertIn("sharpe", result.windows[0])
        self.assertAlmostEqual(result.avg_sharpe, 1.5, places=4)
        self.assertAlmostEqual(result.avg_drawdown, 0.1, places=4)

    @patch('backend.app.backtest.walk_forward.run_backtest')
    def test_insufficient_data(self, mock_run):
        """When there is not enough data for a single window, result should be empty."""
        # Use a very short price series
        short_prices = self.prices.iloc[:100]  # less than train+test
        result = walk_forward(self.dummy_signals, short_prices)

        self.assertEqual(len(result.windows), 0)
        self.assertEqual(result.avg_sharpe, 0.0)
        self.assertEqual(result.avg_drawdown, 0.0)
        mock_run.assert_not_called()

    @patch('backend.app.backtest.walk_forward.run_backtest')
    def test_signal_exception_handling(self, mock_run):
        """If signals_fn raises an exception, the window should capture the error."""
        def faulty_signals(train, test):
            raise ValueError("signal generation failed")

        # Ensure there is enough data for at least one window
        train_bars = TIMEFRAME_TRAIN * 252
        test_bars = TIMEFRAME_TEST * 21
        prices = self.prices.iloc[:train_bars + test_bars]

        result = walk_forward(faulty_signals, prices)

        self.assertEqual(len(result.windows), 1)
        self.assertIn("error", result.windows[0])
        self.assertIn("signal generation failed", result.windows[0]["error"])
        # No backtest should have been executed
        mock_run.assert_not_called()

if __name__ == "__main__":
    unittest.main()