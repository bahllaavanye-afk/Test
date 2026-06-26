"""ML-enhanced momentum: Jegadeesh-Titman signals filtered by LSTM + XGBoost ensemble."""
import logging
from typing import Any

import pandas as pd

from app.ml.inference import get_inference_service
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.momentum import MomentumStrategy

logger = logging.getLogger(__name__)


class MLMomentumStrategy(AbstractStrategy):
    name = "ml_momentum"
    display_name = "ML Momentum (LSTM + XGBoost Filter)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 3600.0
    confidence_threshold = 0.65

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._base = MomentumStrategy(params)

    async def analyze(self, data: pd.DataFrame | None, symbol: str | None) -> Signal | None:
        # Edge‑case handling for inputs
        if data is None or data.empty:
            logger.debug("MLMomentumStrategy.analyze called with None or empty data.")
            return None
        if not symbol:
            logger.debug("MLMomentumStrategy.analyze called with empty symbol.")
            return None

        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # Apply ML filter
        try:
            inference = get_inference_service()
            ml_result: Any = await inference.predict(data, symbol)

            # Defensive checks on ml_result structure
            if not isinstance(ml_result, dict):
                logger.debug("ML result is not a dict: %s", ml_result)
                return None

            prediction = ml_result.get("prediction")
            if prediction not in {"up", "down", "neutral"}:
                logger.debug("Unexpected prediction value: %s", prediction)
                return None

            if prediction == "neutral":
                return None

            ml_confidence = ml_result.get("confidence")
            if not isinstance(ml_confidence, (int, float)):
                logger.debug("Invalid or missing confidence in ML result: %s", ml_confidence)
                ml_confidence = 0.0

            # Only pass if ML agrees with indicator direction
            if prediction == "up" and getattr(base_signal, "side", None) == "buy":
                combined_conf = (getattr(base_signal, "confidence", 0) + ml_confidence) / 2
                base_signal.confidence = min(0.95, combined_conf)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                base_signal.metadata["ml_confidence"] = ml_confidence
                return base_signal
            elif prediction == "down" and getattr(base_signal, "side", None) == "sell":
                combined_conf = (getattr(base_signal, "confidence", 0) + ml_confidence) / 2
                base_signal.confidence = min(0.95, combined_conf)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                base_signal.metadata["ml_confidence"] = ml_confidence
                return base_signal
        except Exception as e:
            logger.exception("Error during ML inference for %s: %s", symbol, e)
            # Fall back to base signal if ML unavailable; returning None keeps current behavior
        return None

    def backtest_signals(self, df: pd.DataFrame | None) -> BacktestSignals:
        # Guard against None or empty DataFrames during backtesting
        if df is None or df.empty:
            logger.debug("Backtest called with None or empty DataFrame.")
            return BacktestSignals(signals=[])
        # For backtesting without live ML: use base signals as proxy
        return self._base.backtest_signals(df)