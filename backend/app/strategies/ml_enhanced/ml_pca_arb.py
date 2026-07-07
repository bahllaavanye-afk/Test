"""
ML-Enhanced PCA Statistical Arbitrage Strategy.

Extends PCAStatArbStrategy by gating entries through an LSTM confidence
filter: a trade is only taken when BOTH conditions are true:

  1. PCA s-score exceeds the entry threshold (mean-reversion signal)
  2. LSTM model confidence > 0.60 (directional agreement)

If the ML inference service is unavailable the strategy falls back
gracefully (returns None from analyze, uses base signals in backtest).
"""

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.pca_stat_arb import PCAStatArbStrategy

# Constants
DEFAULT_ML_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_STRATEGY_CONFIDENCE_THRESHOLD = 0.65
DAILY_TICK_INTERVAL_SECONDS = 86_400.0
MAX_BLENDED_CONFIDENCE = 0.95
PREDICTION_NEUTRAL = "neutral"
PREDICTION_UP = "up"
PREDICTION_DOWN = "down"
SIDE_BUY = "buy"
SIDE_SELL = "sell"
METADATA_ML_CONFIDENCE_KEY = "ml_confidence"
STRATEGY_NAME = "ml_pca_arb"
STRATEGY_DISPLAY_NAME = "ML PCA Statistical Arbitrage (LSTM-Gated)"

# ML inference is optional — import defensively
try:
    from app.ml.inference import get_inference_service as _get_inference_service
    _INFERENCE_AVAILABLE = True
except Exception:
    _INFERENCE_AVAILABLE = False


class MLPCAStatArbStrategy(AbstractStrategy):
    """
    ML-gated PCA Statistical Arbitrage.

    Same s-score logic as PCAStatArbStrategy but each entry signal is
    filtered through an LSTM model.  When the ML service is not loaded
    the strategy degrades gracefully:
      - analyze()           → returns None (no signal)
      - backtest_signals()  → delegates to the base PCA strategy
    """

    name = STRATEGY_NAME
    display_name = STRATEGY_DISPLAY_NAME
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "arbitrage"
    tick_interval_seconds = DAILY_TICK_INTERVAL_SECONDS
    confidence_threshold = DEFAULT_STRATEGY_CONFIDENCE_THRESHOLD

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self._base = PCAStatArbStrategy(params)
        self._ml_threshold: float = float(
            p.get("ml_confidence_threshold", DEFAULT_ML_CONFIDENCE_THRESHOLD)
        )

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate a signal only when PCA s-score AND LSTM agree.

        Falls back to None (no trade) when ML is unavailable.
        """
        # Step 1: get base PCA signal
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # Step 2: apply ML filter
        if not _INFERENCE_AVAILABLE:
            # ML service not installed — skip silently
            return None

        try:
            inference = _get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None:
                return None

            ml_confidence: float = float(ml_result.get("confidence", 0.0))
            ml_prediction: str = ml_result.get("prediction", PREDICTION_NEUTRAL)

            if ml_confidence < self._ml_threshold:
                return None
            if ml_prediction == PREDICTION_NEUTRAL:
                return None

            # Direction agreement check
            direction_ok = (
                (ml_prediction == PREDICTION_UP and base_signal.side == SIDE_BUY)
                or (ml_prediction == PREDICTION_DOWN and base_signal.side == SIDE_SELL)
            )
            if not direction_ok:
                return None

            # Blend confidences
            blended = min(MAX_BLENDED_CONFIDENCE, (base_signal.confidence + ml_confidence) / 2)
            base_signal.confidence = blended
            base_signal.strategy_name = self.name
            base_signal.strategy_type = self.strategy_type
            base_signal.metadata[METADATA_ML_CONFIDENCE_KEY] = ml_confidence
            return base_signal

        except Exception:
            # ML service raised an error — degrade gracefully
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Delegate to the base PCA strategy for backtesting.

        In a production backtest with a trained LSTM available, the signals
        would be gated per-bar.  Without a serialized model this delegation
        is the correct fallback: it still uses the same PCA edge.
        """
        return self._base.backtest_signals(df)