"""ML-enhanced momentum strategy.

This module implements a momentum strategy that combines the classic Jegadeesh‑Titman
signal with a machine‑learning filter based on an LSTM + XGBoost ensemble. The base
momentum logic is provided by :class:`app.strategies.manual.momentum.MomentumStrategy`,
while the ML inference is performed via the shared inference service.

The strategy only emits a signal when both the traditional indicator and the ML model
agree on direction, and it adjusts the confidence accordingly. Additional
confirmation filters tighten entry conditions, and basic exit‑related metadata
(stop‑loss / take‑profit) is attached to the signal.
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

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Create a new ``MLMomentumStrategy`` instance.

        Parameters
        ----------
        params : dict | None, optional
            Optional configuration parameters passed to the base strategy.
        """
        super().__init__(params)
        self._base = MomentumStrategy(params)

        # Tunable filter parameters with sensible defaults
        self.momentum_window = params.get("momentum_window", 20) if params else 20
        self.momentum_threshold = params.get("momentum_threshold", 0.02) if params else 0.02
        self.volume_window = params.get("volume_window", 20) if params else 20
        self.volume_multiplier = params.get("volume_multiplier", 1.2) if params else 1.2

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """Generate a trading signal for a given symbol.

        The method first obtains a signal from the underlying momentum strategy.
        It then applies a series of entry‑confirmation filters before querying the
        ML inference service.  If the ML prediction agrees with the base signal
        direction and the combined confidence exceeds the configured threshold,
        the signal confidence is adjusted and returned.

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
            ML models agree and all filters pass, otherwise ``None``.
        """
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # Entry‑confirmation filters
        if not self._passes_entry_filters(data, base_signal.side):
            return None

        try:
            inference = get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None or ml_result.get("prediction") == "neutral":
                return None

            return self._apply_ml_filter(base_signal, ml_result, data)
        except Exception as e:  # pragma: no cover
            logger.exception("ML inference failed for %s: %s", symbol, e)
            return None

    def _passes_entry_filters(self, data: pd.DataFrame, side: str) -> bool:
        """Validate additional entry conditions.

        The filters include:
        * Momentum magnitude over a configurable window.
        * Volume strength relative to a moving average.
        * Recent price positioned above (for long) or below (for short) its
          simple moving average.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price data containing at least ``close`` and ``volume`` columns.
        side : str
            Desired trade side (``"buy"`` or ``"sell"``).

        Returns
        -------
        bool
            ``True`` if all filters pass, otherwise ``False``.
        """
        if data.empty or not {"close", "volume"}.issubset(data.columns):
            return False

        # Momentum: (current - past) / past
        past_price = data["close"].iloc[-self.momentum_window]
        current_price = data["close"].iloc[-1]
        momentum = (current_price - past_price) / past_price

        if side == "buy" and momentum < self.momentum_threshold:
            return False
        if side == "sell" and momentum > -self.momentum_threshold:
            return False

        # Volume filter: current volume > avg(volume) * multiplier
        avg_volume = data["volume"].iloc[-self.volume_window :].mean()
        current_volume = data["volume"].iloc[-1]
        if current_volume < avg_volume * self.volume_multiplier:
            return False

        # Simple moving average price filter
        sma = data["close"].iloc[-self.momentum_window :].mean()
        if side == "buy" and current_price < sma:
            return False
        if side == "sell" and current_price > sma:
            return False

        return True

    def _apply_ml_filter(self, base_signal: Signal, ml_result: Dict[str, Any], data: pd.DataFrame) -> Optional[Signal]:
        """Adjust the base signal if the ML prediction agrees.

        Parameters
        ----------
        base_signal : Signal
            Signal produced by the underlying momentum strategy.
        ml_result : dict
            Result from the ML inference service containing ``prediction`` and
            ``confidence`` keys.
        data : pd.DataFrame
            Historical data used for supplemental exit‑related metadata.

        Returns
        -------
        Signal | None
            Updated signal if directions match and combined confidence meets the
            configured threshold, otherwise ``None``.
        """
        prediction = ml_result["prediction"]
        ml_conf = ml_result["confidence"]

        side_match = (
            (prediction == "up" and base_signal.side == "buy")
            or (prediction == "down" and base_signal.side == "sell")
        )
        if not side_match:
            return None

        # Combine confidences (simple average) and enforce threshold
        combined_confidence = (base_signal.confidence + ml_conf) / 2
        if combined_confidence < self.confidence_threshold:
            return None

        # Cap combined confidence to avoid over‑confidence
        combined_confidence = min(0.95, combined_confidence)

        # Attach enriched metadata
        base_signal.confidence = combined_confidence
        base_signal.strategy_name = self.name
        base_signal.strategy_type = self.strategy_type
        base_signal.metadata["ml_confidence"] = ml_conf
        base_signal.metadata["momentum"] = (data["close"].iloc[-1] - data["close"].iloc[-self.momentum_window]) / data["close"].iloc[-self.momentum_window]

        # Basic exit‑related levels
        recent_high = data["high"].iloc[-1] if "high" in data.columns else data["close"].iloc[-1]
        recent_low = data["low"].iloc[-1] if "low" in data.columns else data["close"].iloc[-1]
        if base_signal.side == "buy":
            base_signal.metadata["stop_loss"] = recent_low * 0.98
            base_signal.metadata["take_profit"] = recent_high * 1.04
        else:
            base_signal.metadata["stop_loss"] = recent_high * 1.02
            base_signal.metadata["take_profit"] = recent_low * 0.96

        return base_signal

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