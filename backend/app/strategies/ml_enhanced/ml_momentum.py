"""ML-enhanced momentum: Jegadeesh-Titman signals filtered by LSTM + XGBoost ensemble."""
import copy
import logging
from typing import Any

import numpy as np
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
    confidence_threshold = 0.65  # minimum confidence for both ML and base signal

    # Additional filters
    momentum_return_threshold = 0.005  # 0.5% price move
    volatility_min_threshold = 0.001   # minimum std of returns

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._base = MomentumStrategy(params)

    async def analyze(self, data: pd.DataFrame | None, symbol: str | None) -> Signal | None:
        # Guard clauses
        if data is None or data.empty:
            logger.debug("MLMomentumStrategy.analyze called with None or empty data.")
            return None
        if not symbol:
            logger.debug("MLMomentumStrategy.analyze called with empty symbol.")
            return None

        # Base momentum signal
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # Ensure base signal meets a minimal confidence level
        if getattr(base_signal, "confidence", 0.0) < self.confidence_threshold:
            logger.debug(
                "Base signal confidence %.3f below threshold %.3f",
                getattr(base_signal, "confidence", 0.0),
                self.confidence_threshold,
            )
            return None
        # Basic price‑movement filters
        if len(data) >= 2:
            price_change = (data["close"].iloc[-1] - data["close"].iloc[-2]) / data["close"].iloc[-2]
        else:
            price_change = 0.0

        recent_vol = (
            data["close"]
            .pct_change()
            .rolling(window=5)
            .std()
            .iloc[-1]
            if len(data) >= 5
            else 0.0
        )

        if recent_vol < self.volatility_min_threshold:
            logger.debug(
                "Recent volatility %.6f below minimum %.6f; rejecting signal.",
                recent_vol,
                self.volatility_min_threshold,
            )
            return None

        # Apply ML filter
        try:
            inference = get_inference_service()
            ml_result: Any = await inference.predict(data, symbol)

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

            # Require ML confidence to meet the same threshold
            if ml_confidence < self.confidence_threshold:
                logger.debug(
                    "ML confidence %.3f below threshold %.3f; rejecting signal.",
                    ml_confidence,
                    self.confidence_threshold,
                )
                return None

            # Determine alignment between ML and base signal
            base_side = getattr(base_signal, "side", None)

            # Entry logic – both agree and momentum filter satisfied
            if prediction == "up" and base_side == "buy":
                if price_change < self.momentum_return_threshold:
                    logger.debug(
                        "Price change %.4f below upward momentum threshold %.4f",
                        price_change,
                        self.momentum_return_threshold,
                    )
                    return None
                combined_conf = (getattr(base_signal, "confidence", 0.0) + ml_confidence) / 2
                base_signal.confidence = min(0.95, combined_conf)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                base_signal.metadata["ml_confidence"] = ml_confidence
                return base_signal

            if prediction == "down" and base_side == "sell":
                if price_change > -self.momentum_return_threshold:
                    logger.debug(
                        "Price change %.4f above downward momentum threshold %.4f",
                        price_change,
                        -self.momentum_return_threshold,
                    )
                    return None
                combined_conf = (getattr(base_signal, "confidence", 0.0) + ml_confidence) / 2
                base_signal.confidence = min(0.95, combined_conf)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                base_signal.metadata["ml_confidence"] = ml_confidence
                return base_signal

            # Exit/close logic – ML predicts opposite direction with sufficient confidence
            if prediction == "up" and base_side == "sell":
                # Generate a closing signal for an existing short position
                exit_signal = copy.deepcopy(base_signal)
                exit_signal.side = "buy"  # close short
                exit_signal.confidence = min(0.90, ml_confidence)
                exit_signal.strategy_name = self.name
                exit_signal.strategy_type = self.strategy_type
                exit_signal.metadata["ml_confidence"] = ml_confidence
                exit_signal.metadata["exit_reason"] = "ml_reverse"
                return exit_signal

            if prediction == "down" and base_side == "buy":
                # Generate a closing signal for an existing long position
                exit_signal = copy.deepcopy(base_signal)
                exit_signal.side = "sell"  # close long
                exit_signal.confidence = min(0.90, ml_confidence)
                exit_signal.strategy_name = self.name
                exit_signal.strategy_type = self.strategy_type
                exit_signal.metadata["ml_confidence"] = ml_confidence
                exit_signal.metadata["exit_reason"] = "ml_reverse"
                return exit_signal

        except Exception as e:
            logger.exception("Error during ML inference for %s: %s", symbol, e)
            # If ML fails, we simply fall back to no signal (None)

        return None

    def backtest_signals(self, df: pd.DataFrame | None) -> BacktestSignals:
        # Guard against None or empty DataFrames during backtesting
        if df is None or df.empty:
            logger.debug("Backtest called with None or empty DataFrame.")
            return BacktestSignals(signals=[])
        # For backtesting without live ML: use base signals as proxy
        return self._base.backtest_signals(df)