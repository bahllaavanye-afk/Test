"""
ML-Enhanced PCA Statistical Arbitrage Strategy.

Extends PCAStatArbStrategy by gating entries through an LSTM confidence
filter and additional market‑condition confirmations.  A trade is only
taken when ALL conditions are true:

  1. PCA s‑score exceeds the base entry threshold (mean‑reversion signal)
  2. LSTM model confidence > configurable threshold
  3. Recent volume is above the historical median (liquidity filter)

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


# Default thresholds – can be overridden via strategy parameters
_DEFAULT_ML_CONFIDENCE_THRESHOLD = 0.60
_DEFAULT_ML_EXIT_CONFIDENCE_THRESHOLD = 0.40
_DEFAULT_SCORE_ENTRY_THRESHOLD = 2.0
_DEFAULT_VOLUME_MULTIPLIER = 1.0  # require recent volume >= median * multiplier


class MLPCAStatArbStrategy(AbstractStrategy):
    """
    ML‑gated PCA Statistical Arbitrage.

    Mirrors the s‑score logic of :class:`PCAStatArbStrategy` but each
    entry signal is filtered through an LSTM model and additional
    confirmation checks.  When the ML service is not loaded the strategy
    degrades gracefully:
      - ``analyze()`` → returns ``None`` (no signal)
      - ``backtest_signals()`` → delegates to the base PCA strategy
    """

    name = "ml_pca_arb"
    display_name = "ML PCA Statistical Arbitrage (LSTM‑Gated)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 86_400.0  # daily

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self._base = PCAStatArbStrategy(params)

        # Configurable thresholds
        self._ml_entry_threshold: float = float(
            p.get("ml_confidence_threshold", _DEFAULT_ML_CONFIDENCE_THRESHOLD)
        )
        self._ml_exit_threshold: float = float(
            p.get("ml_exit_confidence_threshold", _DEFAULT_ML_EXIT_CONFIDENCE_THRESHOLD)
        )
        self._score_entry_threshold: float = float(
            p.get("score_entry_threshold", _DEFAULT_SCORE_ENTRY_THRESHOLD)
        )
        self._volume_multiplier: float = float(
            p.get("volume_multiplier", _DEFAULT_VOLUME_MULTIPLIER)
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _volume_ok(self, data: pd.DataFrame) -> bool:
        """Return True if recent volume is at least ``median * multiplier``."""
        if "volume" not in data.columns:
            return True  # cannot evaluate – assume ok
        recent_vol = data["volume"].tail(5).mean()
        median_vol = data["volume"].median()
        return recent_vol >= median_vol * self._volume_multiplier

    def _entry_filters_pass(self, base_signal: Signal, ml_confidence: float) -> bool:
        """Apply all entry‑side confirmation filters."""
        # 1. Base confidence must be reasonable
        if base_signal.confidence < 0.5:
            return False

        # 2. s‑score magnitude check
        s_score = base_signal.metadata.get("s_score")
        if s_score is not None and abs(s_score) < self._score_entry_threshold:
            return False

        # 3. Volume filter – caller supplies data context
        # (Handled separately in ``analyze`` where data is available)

        # 4. ML confidence threshold
        if ml_confidence < self._ml_entry_threshold:
            return False

        return True

    def _exit_filters_pass(self, base_signal: Signal, ml_confidence: float) -> bool:
        """Apply exit‑side filters – looser than entry."""
        # Allow exit if confidence falls below exit threshold
        return ml_confidence <= self._ml_exit_threshold

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate a signal when PCA s‑score and LSTM agree, plus confirmation
        filters.  If the ML service is unavailable or any filter fails, ``None``
        is returned, resulting in no trade.
        """
        # Step 1: obtain the base PCA signal
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # Step 2: if ML inference is unavailable, fall back to no signal
        if not _INFERENCE_AVAILABLE:
            return None

        try:
            inference = _get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None:
                return None

            ml_confidence: float = float(ml_result.get("confidence", 0.0))
            ml_prediction: str = ml_result.get("prediction", "neutral")

            # Apply volume confirmation early – cheap check
            if not self._volume_ok(data):
                return None

            # ENTRY path
            if base_signal.side in ("buy", "sell"):
                if ml_prediction == "neutral":
                    return None
                direction_ok = (
                    (ml_prediction == "up" and base_signal.side == "buy")
                    or (ml_prediction == "down" and base_signal.side == "sell")
                )
                if not direction_ok:
                    return None

                if not self._entry_filters_pass(base_signal, ml_confidence):
                    return None

                # Blend confidences – cap to avoid over‑confidence
                blended = min(0.95, (base_signal.confidence + ml_confidence) / 2)
                base_signal.confidence = blended
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                base_signal.metadata["ml_confidence"] = ml_confidence
                base_signal.metadata["ml_prediction"] = ml_prediction
                return base_signal

            # EXIT path – assume base_signal.side == "exit" or similar
            # We still respect ML direction; if confidence drops below exit
            # threshold we allow the exit to proceed.
            if ml_prediction != "neutral" and self._exit_filters_pass(base_signal, ml_confidence):
                base_signal.confidence = min(base_signal.confidence, ml_confidence)
                base_signal.strategy_name = self.name
                base_signal.strategy_type = self.strategy_type
                base_signal.metadata["ml_confidence"] = ml_confidence
                base_signal.metadata["ml_prediction"] = ml_prediction
                return base_signal

            return None

        except Exception:
            # Any error from the ML service results in graceful degradation.
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Delegate to the base PCA strategy for backtesting.

        In a production backtest with a trained LSTM available, the signals
        would be gated per‑bar.  Without a serialized model this delegation
        is the correct fallback: it still uses the same PCA edge.
        """
        return self._base.backtest_signals(df)