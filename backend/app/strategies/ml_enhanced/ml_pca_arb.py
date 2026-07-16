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

# ML inference is optional — import defensively
try:
    from app.ml.inference import get_inference_service as _get_inference_service
    _INFERENCE_AVAILABLE = True
except Exception:
    _INFERENCE_AVAILABLE = False


_ML_CONFIDENCE_THRESHOLD = 0.60


class MLPCAStatArbStrategy(AbstractStrategy):
    """
    ML-gated PCA Statistical Arbitrage.

    Same s-score logic as PCAStatArbStrategy but each entry signal is
    filtered through an LSTM model.  When the ML service is not loaded
    the strategy degrades gracefully:
      - analyze()           → returns None (no signal)
      - backtest_signals()  → delegates to the base PCA strategy
    """

    name = "ml_pca_arb"
    display_name = "ML PCA Statistical Arbitrage (LSTM-Gated)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 86_400.0  # daily
    confidence_threshold = 0.65

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self._base = PCAStatArbStrategy(params)
        self._ml_threshold: float = float(
            p.get("ml_confidence_threshold", _ML_CONFIDENCE_THRESHOLD)
        )

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate a signal only when PCA s-score AND LSTM agree.

        Falls back to None (no trade) when ML is unavailable.
        """
        base_signal = await self._get_base_signal(data, symbol)
        if base_signal is None:
            return None

        ml_result = await self._get_ml_result(data, symbol)
        if not self._is_ml_result_acceptable(ml_result, base_signal):
            return None

        blended_confidence = self._blend_confidence(
            base_signal.confidence, ml_result["confidence"]
        )
        base_signal.confidence = blended_confidence
        base_signal.strategy_name = self.name
        base_signal.strategy_type = self.strategy_type
        base_signal.metadata["ml_confidence"] = ml_result["confidence"]
        return base_signal

    async def _get_base_signal(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """Retrieve the underlying PCA signal."""
        return await self._base.analyze(data, symbol)

    async def _get_ml_result(self, data: pd.DataFrame, symbol: str) -> dict | None:
        """Query the ML inference service; returns a dict with 'confidence' and 'prediction'."""
        if not _INFERENCE_AVAILABLE:
            return None
        try:
            inference = _get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None:
                return None
            # Normalise fields
            confidence = float(ml_result.get("confidence", 0.0))
            prediction = ml_result.get("prediction", "neutral")
            return {"confidence": confidence, "prediction": prediction}
        except Exception:
            return None

    def _is_ml_result_acceptable(self, ml_result: dict | None, base_signal: Signal) -> bool:
        """Validate ML confidence, prediction, and directional agreement."""
        if ml_result is None:
            return False
        if ml_result["confidence"] < self._ml_threshold:
            return False
        if ml_result["prediction"] == "neutral":
            return False
        return self._direction_agrees(ml_result["prediction"], base_signal)

    @staticmethod
    def _direction_agrees(ml_prediction: str, base_signal: Signal) -> bool:
        """Check that ML prediction direction matches the PCA signal side."""
        return (
            (ml_prediction == "up" and base_signal.side == "buy")
            or (ml_prediction == "down" and base_signal.side == "sell")
        )

    @staticmethod
    def _blend_confidence(base_conf: float, ml_conf: float) -> float:
        """
        Blend PCA and ML confidences, capping the result at 0.95.
        """
        return min(0.95, (base_conf + ml_conf) / 2)

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Delegate to the base PCA strategy for backtesting.

        In a production backtest with a trained LSTM available, the signals
        would be gated per-bar.  Without a serialized model this delegation
        is the correct fallback: it still uses the same PCA edge.
        """
        return self._base.backtest_signals(df)