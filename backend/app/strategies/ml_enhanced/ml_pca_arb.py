"""
ML-Enhanced PCA Statistical Arbitrage Strategy.

Extends PCAStatArbStrategy by gating entries through an LSTM confidence
filter: a trade is only taken when BOTH conditions are true:

  1. PCA s-score exceeds the entry threshold (mean-reversion signal)
  2. LSTM model confidence > 0.60 (directional agreement)

If the ML inference service is unavailable the strategy falls back
gracefully (returns None from analyze, uses base signals in backtest).
"""
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

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        p = params or {}
        self._base = PCAStatArbStrategy(params)
        self._ml_threshold: float = float(
            p.get("ml_confidence_threshold", _ML_CONFIDENCE_THRESHOLD)
        )
        # Monitoring metrics
        self._signal_count: int = 0
        self._cumulative_pnl: float = 0.0

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate a signal only when PCA s-score AND LSTM agree.

        Falls back to None (no trade) when ML is unavailable.
        """
        start_time = time.perf_counter()
        signal: Signal | None = None

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

            signal = base_signal
            # Update monitoring counters
            self._signal_count += 1
            # If the signal carries an expected P&L, accumulate it; otherwise ignore
            pnl = getattr(signal, "expected_pnl", None)
            if isinstance(pnl, (int, float)):
                self._cumulative_pnl += pnl

            return signal

        except Exception:
            # ML service raised an error — degrade gracefully
            return None

        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _logger.info(
                "MLPCAStatArb analyze completed",
                extra={
                    "symbol": symbol,
                    "execution_time_ms": round(elapsed_ms, 2),
                    "signal_generated": signal is not None,
                    "signal_count": self._signal_count,
                    "cumulative_pnl": round(self._cumulative_pnl, 4),
                },
            )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Delegate to the base PCA strategy for backtesting.

        In a production backtest with a trained LSTM available, the signals
        would be gated per-bar.  Without a serialized model this delegation
        is the correct fallback: it still uses the same PCA edge.
        """
        start_time = time.perf_counter()
        try:
            result = self._base.backtest_signals(df)
            return result
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            _logger.info(
                "MLPCAStatArb backtest_signals executed",
                extra={
                    "execution_time_ms": round(elapsed_ms, 2),
                    "signal_count": self._signal_count,
                    "cumulative_pnl": round(self._cumulative_pnl, 4),
                },
            )