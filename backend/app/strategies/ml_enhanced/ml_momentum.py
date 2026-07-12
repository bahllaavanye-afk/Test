"""ML-enhanced momentum strategy.

This module implements a momentum strategy that combines the classic Jegadeesh‑Titman
signal with a machine‑learning filter based on an LSTM + XGBoost ensemble. The base
momentum logic is provided by :class:`app.strategies.manual.momentum.MomentumStrategy`,
while the ML inference is performed via the shared inference service.

The strategy only emits a signal when both the traditional indicator and the ML model
agree on direction, and it adjusts the confidence accordingly. Additional entry
filters tighten the signal quality, and an exit condition is generated when the ML
model predicts an opposite direction with sufficient confidence.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.momentum import MomentumStrategy
from app.ml.inference import get_inference_service


logger = logging.getLogger(__name__)


class MLMomentumStrategy(AbstractStrategy):
    """ML‑enhanced momentum strategy.

    The strategy wraps the classic momentum logic and applies an ML filter,
    adding tighter entry filters and a simple ML‑driven exit condition.
    """

    name = "ml_momentum"
    display_name = "ML Momentum (LSTM + XGBoost Filter)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.65

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Create a new ``MLMomentumStrategy`` instance.

        Parameters
        ----------
        params : dict | None, optional
            Optional configuration parameters passed to the base strategy.
        """
        super().__init__(params)
        self._base = MomentumStrategy(params)
        # Track open positions per symbol for simple exit handling.
        self._open_positions: Dict[str, Signal] = {}

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate a trading signal for a given symbol.

        The method first obtains a signal from the underlying momentum strategy.
        If a base signal is present, it applies entry filters before querying the
        ML inference service.  An ML‑driven exit is emitted when the model predicts
        the opposite direction with sufficient confidence.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price and indicator data for the symbol.
        symbol : str
            Ticker symbol for which the signal is being generated.

        Returns
        -------
        Signal | None
            A populated :class:`app.strategies.base.Signal` if conditions are met,
            otherwise ``None``.
        """
        # Attempt to get the base momentum signal (may be ``None``).
        base_signal = await self._base.analyze(data, symbol)

        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)

            # If the ML model cannot produce a directional prediction, stop here.
            if ml_result is None or ml_result.get("prediction") == "neutral":
                return None

            # Check for an existing open position – generate an exit if ML predicts
            # the opposite direction with confidence above the threshold.
            open_signal = self._open_positions.get(symbol)
            if open_signal and self._should_exit(open_signal, ml_result):
                exit_signal = self._create_exit_signal(symbol, ml_result)
                self._open_positions.pop(symbol, None)
                return exit_signal

            # No base signal → no entry; exit logic already handled above.
            if base_signal is None:
                return None

            # Apply tightened entry filters before accepting the ML agreement.
            if not self._entry_filters_pass(data, base_signal):
                return None

            # Apply the ML filter; if it returns ``None`` the entry is rejected.
            filtered_signal = self._apply_ml_filter(base_signal, ml_result)
            if filtered_signal:
                self._open_positions[symbol] = filtered_signal
            return filtered_signal

        except Exception as e:  # pragma: no cover
            logger.exception("ML inference failed for %s: %s", symbol, e)
            return None

    def _entry_filters_pass(self, data: pd.DataFrame, base_signal: Signal) -> bool:
        """Run additional entry filters to tighten signal quality.

        Filters include:
        * Minimum base confidence.
        * Recent price momentum consistency.
        * Minimum average volume (if volume data is available).

        Returns ``True`` when all filters are satisfied.
        """
        # Minimum confidence from the underlying momentum strategy.
        base_conf_thresh = (
            self.params.get("base_confidence_threshold", 0.6)
            if self.params
            else 0.6
        )
        if getattr(base_signal, "confidence", 0) < base_conf_thresh:
            return False

        # Ensure enough data points for momentum calculations.
        if len(data) < 6:
            return False

        # Recent price momentum (5‑period price change).
        price_change = (
            data["close"].iloc[-1] - data["close"].iloc[-5]
        ) / data["close"].iloc[-5]

        momentum_min = (
            self.params.get("price_momentum_min", 0.01)
            if self.params
            else 0.01
        )
        if base_signal.side == "buy" and price_change < momentum_min:
            return False
        if base_signal.side == "sell" and price_change > -momentum_min:
            return False

        # Volume filter (optional).
        if "volume" in data.columns:
            avg_vol = data["volume"].iloc[-5:].mean()
            min_vol = (
                self.params.get("min_avg_volume", 0)
                if self.params
                else 0
            )
            if avg_vol < min_vol:
                return False

        return True

    def _apply_ml_filter(self, base_signal: Signal, ml_result: Dict[str, Any]) -> Optional[Signal]:
        """Adjust the base signal if the ML prediction agrees.

        The confidence is combined and capped, and metadata is enriched with the
        ML confidence value.
        """
        prediction = ml_result["prediction"]
        ml_conf = ml_result["confidence"]

        side_match = (
            (prediction == "up" and base_signal.side == "buy")
            or (prediction == "down" and base_signal.side == "sell")
        )
        if not side_match:
            return None

        # Combine confidences, respecting the configured maximum.
        combined_confidence = min(0.95, (base_signal.confidence + ml_conf) / 2)
        base_signal.confidence = combined_confidence
        base_signal.strategy_name = self.name
        base_signal.strategy_type = self.strategy_type
        base_signal.metadata["ml_confidence"] = ml_conf
        return base_signal

    def _should_exit(self, open_signal: Signal, ml_result: Dict[str, Any]) -> bool:
        """Determine whether an open position should be exited."""
        prediction = ml_result["prediction"]
        ml_conf = ml_result["confidence"]
        opposite = (
            (prediction == "down" and open_signal.side == "buy")
            or (prediction == "up" and open_signal.side == "sell")
        )
        return opposite and ml_conf >= self.confidence_threshold

    def _create_exit_signal(self, symbol: str, ml_result: Dict[str, Any]) -> Signal:
        """Create a signal that represents exiting an existing position."""
        exit_side = "sell" if ml_result["prediction"] == "down" else "buy"
        exit_confidence = ml_result["confidence"]
        exit_signal = Signal(
            side=exit_side,
            confidence=exit_confidence,
            metadata={"ml_confidence": exit_confidence, "exit_reason": "ml_opposite"},
        )
        exit_signal.strategy_name = self.name
        exit_signal.strategy_type = self.strategy_type
        return exit_signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Generate back‑test signals for a dataframe.

        For back‑testing environments where live ML inference is unavailable,
        this method falls back to the base momentum signals.
        """
        return self._base.backtest_signals(df)