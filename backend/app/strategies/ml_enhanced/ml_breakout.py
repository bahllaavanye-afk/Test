"""ML-filtered breakout strategy.

This module defines :class:`MLBreakoutStrategy`, an enhanced breakout strategy that
combines the classic breakout logic from :class:`BreakoutStrategy` with a machine‑learning
prediction service. The ML component is used to adjust the confidence of the base
signal when the model predicts a strong upward movement.
"""

import pandas as pd
from typing import Any, Dict, Optional

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.breakout import BreakoutStrategy
from app.ml.inference import get_inference_service


class MLBreakoutStrategy(AbstractStrategy):
    """ML‑enhanced breakout strategy.

    The strategy first runs the traditional breakout analysis and, if a signal is
    generated, queries an external ML inference service. When the ML model predicts
    an upward move with sufficient confidence, the base signal's confidence is
    adjusted and the signal metadata is updated to reflect the ML‑enhanced strategy.
    """

    name = "ml_breakout"
    display_name = "ML Breakout (Volume + Ensemble)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 900.0

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Create a new instance of :class:`MLBreakoutStrategy`.

        Args:
            params: Optional dictionary of configuration parameters passed to the
                underlying :class:`BreakoutStrategy`. If ``None``, defaults are used.
        """
        super().__init__(params)
        self._base = BreakoutStrategy(params)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate a trading signal based on breakout and ML inference.

        The method first obtains a signal from the base breakout strategy. If that
        signal exists, it queries the ML inference service. When the ML result has
        a confidence greater than ``0.65`` and predicts an upward movement, the
        base signal's confidence is updated and the signal is marked as coming
        from this ML‑enhanced strategy.

        Args:
            data: Historical price and volume data for the symbol.
            symbol: Ticker symbol of the asset being analyzed.

        Returns:
            An updated :class:`Signal` instance if the combined criteria are met,
            the original base signal if an exception occurs during ML inference,
            or ``None`` if no suitable signal is produced.
        """
        base_signal = await self._base.analyze(data, symbol)
        if not base_signal:
            return None
        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if (
                ml_result
                and ml_result["confidence"] > 0.65
                and ml_result["prediction"] == "up"
            ):
                base_signal.confidence = min(
                    0.92, (base_signal.confidence + ml_result["confidence"]) / 2
                )
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                return base_signal
        except Exception:
            return base_signal
        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Delegate backtesting to the underlying breakout strategy.

        Args:
            df: DataFrame containing historical data for backtesting.

        Returns:
            A :class:`BacktestSignals` object produced by the base breakout strategy.
        """
        return self._base.backtest_signals(df)