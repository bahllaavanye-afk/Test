"""ML-enhanced momentum strategy.

This module implements a momentum strategy that combines the classic Jegadeesh‑Titman
signal with a machine‑learning filter based on an LSTM + XGBoost ensemble. The base
momentum logic is provided by :class:`app.strategies.manual.momentum.MomentumStrategy`,
while the ML inference is performed via the shared inference service.

The strategy only emits a signal when both the traditional indicator and the ML model
agree on direction, and it tightens entry conditions with additional confirmation
filters. Exit logic is also provided to close positions when the underlying
momentum reverses or the ML model signals an opposite direction.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.strategies.manual.momentum import MomentumStrategy
from app.ml.inference import get_inference_service


logger = logging.getLogger(__name__)


class MLMomentumStrategy(AbstractStrategy):
    """ML‑enhanced momentum strategy.

    The strategy wraps the classic momentum logic and applies an ML filter.
    It inherits from :class:`app.strategies.base.AbstractStrategy`.
    """

    name = "ml_momentum"
    display_name = "ML Momentum (LSTM + XGBoost Filter)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.65

    # Parameters for additional confirmation filters
    momentum_lookback = 20          # periods for raw momentum calculation
    momentum_min_threshold = 0.02  # 2% raw momentum required for entry
    volume_multiplier = 1.5        # volume surge factor relative to median

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Create a new ``MLMomentumStrategy`` instance.

        Parameters
        ----------
        params : dict | None, optional
            Optional configuration parameters passed to the base strategy.
        """
        super().__init__(params)
        self._base = MomentumStrategy(params)

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate a trading signal for a given symbol.

        The method first obtains a signal from the underlying momentum strategy.
        If a base signal is present, it applies additional raw‑momentum and volume
        confirmation filters before querying the ML inference service.  When the
        ML prediction agrees with the base signal direction and the confidence
        exceeds the configured threshold, the signal confidence is adjusted and
        returned.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price and indicator data for the symbol.
        symbol : str
            Ticker symbol for which the signal is being generated.

        Returns
        -------
        Signal | None
            A populated :class:`app.strategies.base.Signal` if both the base and
            ML models agree and confirmation filters pass, otherwise ``None``.
        """
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # Apply raw‑momentum and volume confirmation filters
        if not self._passes_confirmation_filters(data, base_signal.side):
            logger.debug("%s: confirmation filters rejected entry for %s", self.name, symbol)
            return None

        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None or ml_result.get("prediction") == "neutral":
                return None

            return self._apply_ml_filter(base_signal, ml_result)
        except Exception as e:  # pragma: no cover
            logger.exception("ML inference failed for %s: %s", symbol, e)
            return None

    def _passes_confirmation_filters(self, data: pd.DataFrame, side: str) -> bool:
        """Check additional entry filters based on raw momentum and volume.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price data containing at least ``close`` and ``volume`` columns.
        side : str
            Desired trade side from the base strategy (``"buy"`` or ``"sell"``).

        Returns
        -------
        bool
            ``True`` if both momentum and volume conditions are satisfied.
        """
        if "close" not in data.columns or "volume" not in data.columns:
            logger.warning("%s: required columns missing for confirmation filters", self.name)
            return False

        # Raw momentum: percentage change over the lookback window
        if len(data) < self.momentum_lookback:
            return False
        recent_close = data["close"].iloc[-1]
        past_close = data["close"].iloc[-self.momentum_lookback]
        momentum = (recent_close - past_close) / past_close

        # For a long entry we need positive momentum, for short a negative one
        momentum_ok = (momentum >= self.momentum_min_threshold) if side == "buy" else (momentum <= -self.momentum_min_threshold)

        # Volume surge: latest volume vs median of lookback window
        recent_volume = data["volume"].iloc[-1]
        median_volume = data["volume"].iloc[-self.momentum_lookback:].median()
        volume_ok = recent_volume >= self.volume_multiplier * median_volume

        return bool(momentum_ok and volume_ok)

    def _apply_ml_filter(self, base_signal: Signal, ml_result: Dict[str, Any]) -> Optional[Signal]:
        """Adjust the base signal if the ML prediction agrees.

        Parameters
        ----------
        base_signal : Signal
            Signal produced by the underlying momentum strategy.
        ml_result : dict
            Result from the ML inference service containing ``prediction`` and
            ``confidence`` keys.

        Returns
        -------
        Signal | None
            Updated signal if directions match and confidence meets the threshold,
            otherwise ``None``.
        """
        prediction = ml_result["prediction"]
        ml_conf = ml_result["confidence"]

        side_match = (
            (prediction == "up" and base_signal.side == "buy")
            or (prediction == "down" and base_signal.side == "sell")
        )
        if not side_match:
            return None

        # Require both confidences to meet the global threshold
        if ml_conf < self.confidence_threshold or getattr(base_signal, "confidence", 0) < self.confidence_threshold:
            return None

        # Combine confidences with a weighted average favoring the ML model
        combined_confidence = min(0.95, 0.4 * getattr(base_signal, "confidence", 0) + 0.6 * ml_conf)
        base_signal.confidence = combined_confidence
        base_signal.strategy_name = self.name
        base_signal.strategy_type = self.strategy_type
        base_signal.metadata["ml_confidence"] = ml_conf
        base_signal.metadata["combined_confidence"] = combined_confidence
        return base_signal

    def exit_signal(self, data: pd.DataFrame, current_side: str) -> Optional[Signal]:
        """Generate an exit signal based on reversed momentum or ML disagreement.

        Parameters
        ----------
        data : pd.DataFrame
            Recent market data for the instrument.
        current_side : str
            The side of the open position (``"buy"`` or ``"sell"``).

        Returns
        -------
        Signal | None
            A signal indicating the position should be closed, or ``None`` if
            the position should be maintained.
        """
        # Reverse momentum check
        if not self._passes_confirmation_filters(data, current_side):
            logger.debug("%s: momentum reversal detected, exiting %s position", self.name, current_side)
            return Signal(
                side="sell" if current_side == "buy" else "buy",
                confidence=0.9,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                metadata={"reason": "momentum_reversal"},
            )

        # ML disagreement check
        try:
            inference = get_inference_service()
            ml_result = inference.predict_sync(data)  # synchronous fallback for exit checks
            if ml_result is None:
                return None
            prediction = ml_result["prediction"]
            opposite = (prediction == "down" and current_side == "buy") or (prediction == "up" and current_side == "sell")
            if opposite:
                logger.debug("%s: ML disagreement detected, exiting %s position", self.name, current_side)
                return Signal(
                    side="sell" if current_side == "buy" else "buy",
                    confidence=ml_result["confidence"],
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    metadata={"reason": "ml_disagreement"},
                )
        except Exception as e:  # pragma: no cover
            logger.exception("ML exit inference failed: %s", e)

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """Generate back‑test signals for a dataframe.

        For back‑testing environments where live ML inference is unavailable,
        this method falls back to the base momentum signals.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing historical data for back‑testing.

        Returns
        -------
        BacktestSignals
            Signals suitable for back‑testing consumption.
        """
        return self._base.backtest_signals(df)