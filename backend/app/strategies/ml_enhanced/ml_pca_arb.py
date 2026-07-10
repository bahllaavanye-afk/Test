import logging
import time
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

_logger = logging.getLogger(__name__)


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

    _signal_counter: int = 0

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
        start_time = time.monotonic()
        try:
            # Step 1: get base PCA signal
            base_signal = await self._base.analyze(data, symbol)
            if base_signal is None:
                return None

            # Step 2: apply ML filter
            if not _INFERENCE_AVAILABLE:
                # ML service not installed — skip silently
                return None

            inference = _get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if ml_result is None:
                return None

            ml_confidence: float = float(ml_result.get("confidence", 0.0))
            ml_prediction: str = ml_result.get("prediction", "neutral")

            if ml_confidence < self._ml_threshold:
                return None
            if ml_prediction == "neutral":
                return None

            # Direction agreement check
            direction_ok = (
                (ml_prediction == "up" and base_signal.side == "buy")
                or (ml_prediction == "down" and base_signal.side == "sell")
            )
            if not direction_ok:
                return None

            # Blend confidences
            blended = min(0.95, (base_signal.confidence + ml_confidence) / 2)
            base_signal.confidence = blended
            base_signal.strategy_name = self.name
            base_signal.strategy_type = self.strategy_type
            base_signal.metadata["ml_confidence"] = ml_confidence

            # Monitoring: increment counter and log key metrics
            self.__class__._signal_counter += 1
            elapsed_ms = (time.monotonic() - start_time) * 1000
            _logger.info(
                "ml_pca_signal_generated",
                extra={
                    "symbol": symbol,
                    "signal_side": base_signal.side,
                    "ml_confidence": ml_confidence,
                    "blended_confidence": blended,
                    "signal_count": self.__class__._signal_counter,
                    "execution_time_ms": round(elapsed_ms, 2),
                },
            )
            return base_signal
        except Exception:
            # ML service raised an error — degrade gracefully
            return None
        finally:
            # Ensure execution time is logged even when no signal is emitted
            if _logger.isEnabledFor(logging.INFO):
                elapsed_ms = (time.monotonic() - start_time) * 1000
                _logger.info(
                    "ml_pca_analyze_complete",
                    extra={
                        "symbol": symbol,
                        "signal_generated": base_signal is not None,
                        "execution_time_ms": round(elapsed_ms, 2),
                    },
                )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Delegate to the base PCA strategy for backtesting.

        In a production backtest with a trained LSTM available, the signals
        would be gated per-bar.  Without a serialized model this delegation
        is the correct fallback: it still uses the same PCA edge.
        """
        signals = self._base.backtest_signals(df)
        _logger.info(
            "ml_pca_backtest_completed",
            extra={"signal_count": len(signals.signals) if hasattr(signals, "signals") else 0},
        )
        return signals