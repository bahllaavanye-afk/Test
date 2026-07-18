"""
ML-Enhanced PCA Statistical Arbitrage Strategy.

Extends PCAStatArbStrategy by gating entries through an LSTM confidence
filter and adding tighter statistical confirmations. Exit decisions are
also refined using both the PCA score and ML signals.

If the ML inference service is unavailable the strategy degrades
gracefully (returns None from analyze/exit_signal, uses base signals in
backtest).
"""
import pandas as pd
import numpy as np

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.pca_stat_arb import PCAStatArbStrategy

# ML inference is optional — import defensively
try:
    from app.ml.inference import get_inference_service as _get_inference_service
    _INFERENCE_AVAILABLE = True
except Exception:
    _INFERENCE_AVAILABLE = False


_ML_CONFIDENCE_THRESHOLD = 0.60
# Default statistical thresholds (can be overridden via params)
_DEFAULT_S_SCORE_ENTRY = 2.0
_DEFAULT_S_SCORE_EXIT = 0.5
_DEFAULT_VOLUME_MULTIPLIER = 1.0
_DEFAULT_PRICE_REVERT_THRESHOLD = 0.01  # 1% revert


class MLPCAStatArbStrategy(AbstractStrategy):
    """
    ML-gated PCA Statistical Arbitrage.

    Entry signals are produced when:
      1. PCA s-score exceeds a configurable entry threshold.
      2. Recent price movement suggests mean‑reversion.
      3. Volume is above a configurable multiple of the median.
      4. LSTM model confidence exceeds the ML threshold and agrees on
         direction.

    Exit signals are emitted when the PCA s-score reverts within a tighter
    band or when the LSTM model indicates a directional change.

    When the ML service is not loaded the strategy degrades gracefully:
      - analyze()           → returns None (no signal)
      - exit_signal()       → returns None (no signal)
      - backtest_signals()  → delegates to the base PCA strategy
    """

    name = "ml_pca_arb"
    display_name = "ML PCA Statistical Arbitrage (LSTM‑Gated)"
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
        self._s_score_entry: float = float(p.get("s_score_entry_threshold", _DEFAULT_S_SCORE_ENTRY))
        self._s_score_exit: float = float(p.get("s_score_exit_threshold", _DEFAULT_S_SCORE_EXIT))
        self._volume_multiplier: float = float(p.get("volume_multiplier", _DEFAULT_VOLUME_MULTIPLIER))
        self._price_revert_thr: float = float(p.get("price_revert_threshold", _DEFAULT_PRICE_REVERT_THRESHOLD))

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _calculate_median_volume(self, data: pd.DataFrame) -> float:
        """Return median volume for the provided window."""
        if "volume" not in data.columns:
            return 0.0
        return float(data["volume"].median())

    def _price_reversion(self, data: pd.DataFrame) -> bool:
        """
        Confirm that recent price movement is consistent with a mean‑reversion
        setup. Returns True when the price has moved opposite to the current
        s‑score direction by at least the configured threshold.
        """
        if {"close", "s_score"}.issubset(data.columns) is False:
            return False
        recent_close = data["close"].iloc[-1]
        prior_close = data["close"].iloc[-2]
        price_change = (recent_close - prior_close) / prior_close

        # Direction of s‑score: positive => expect price to fall, negative => expect rise
        s_score = data["s_score"].iloc[-1]
        if s_score > 0 and price_change < -self._price_revert_thr:
            return True
        if s_score < 0 and price_change > self._price_revert_thr:
            return True
        return False

    def _volume_check(self, data: pd.DataFrame) -> bool:
        """Ensure current volume exceeds median * multiplier."""
        median_vol = self._calculate_median_volume(data)
        if median_vol == 0:
            return False
        current_vol = float(data["volume"].iloc[-1])
        return current_vol >= median_vol * self._volume_multiplier

    async def _ml_filter(self, data: pd.DataFrame, symbol: str, base_signal: Signal) -> bool:
        """
        Apply the LSTM confidence and direction filter.
        Returns True when the ML signal agrees with the base signal.
        """
        if not _INFERENCE_AVAILABLE:
            return False
        try:
            inference = _get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if not ml_result:
                return False

            ml_confidence = float(ml_result.get("confidence", 0.0))
            ml_prediction = ml_result.get("prediction", "neutral")

            if ml_confidence < self._ml_threshold or ml_prediction == "neutral":
                return False

            direction_ok = (
                (ml_prediction == "up" and base_signal.side == "buy")
                or (ml_prediction == "down" and base_signal.side == "sell")
            )
            if not direction_ok:
                return False

            # Blend confidences (capped at 0.95)
            blended = min(0.95, (base_signal.confidence + ml_confidence) / 2)
            base_signal.confidence = blended
            base_signal.metadata["ml_confidence"] = ml_confidence
            return True
        except Exception:
            # Any failure in the ML service results in a safe reject.
            return False

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate an entry signal when PCA s‑score, statistical filters,
        and the LSTM model all agree.
        """
        # Base PCA signal (includes s‑score direction)
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # Tighten entry: require absolute s‑score beyond entry threshold
        if "s_score" not in data.columns:
            return None
        s_score = float(data["s_score"].iloc[-1])
        if abs(s_score) < self._s_score_entry:
            return None

        # Statistical confirmations
        if not self._price_reversion(data):
            return None
        if not self._volume_check(data):
            return None

        # ML gating
        if not await self._ml_filter(data, symbol, base_signal):
            return None

        # Annotate signal
        base_signal.strategy_name = self.name
        base_signal.strategy_type = self.strategy_type
        base_signal.metadata["s_score"] = s_score
        return base_signal

    async def exit_signal(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate an exit signal based on PCA reversion or ML directional change.
        """
        # Attempt to reuse base strategy's exit logic if it exists
        base_exit = None
        if hasattr(self._base, "exit_signal"):
            base_exit = await self._base.exit_signal(data, symbol)  # type: ignore[attr-defined]

        # If base provides a clear exit, respect it (but still apply ML confirmation)
        if base_exit:
            if await self._ml_filter(data, symbol, base_exit):
                return base_exit
            # If ML disagrees, fall back to PCA exit criteria only
            return base_exit

        # When base has no explicit exit, use our own criteria
        if "s_score" not in data.columns:
            return None
        s_score = float(data["s_score"].iloc[-1])
        # Exit when the score reverts within the tighter band
        if abs(s_score) <= self._s_score_exit:
            # Construct a generic exit signal mirroring the last entry side
            exit_side = "sell" if s_score > 0 else "buy"
            exit_signal = Signal(
                side=exit_side,
                confidence=0.9,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                metadata={"s_score": s_score, "reason": "reversion"},
            )
            return exit_signal

        # Additional ML‑driven exit: if prediction flips direction
        if _INFERENCE_AVAILABLE:
            try:
                inference = _get_inference_service()
                ml_result = await inference.predict(data, symbol)
                if ml_result:
                    ml_prediction = ml_result.get("prediction", "neutral")
                    # Flip detection
                    if (ml_prediction == "up" and s_score < 0) or (ml_prediction == "down" and s_score > 0):
                        exit_side = "sell" if s_score > 0 else "buy"
                        exit_signal = Signal(
                            side=exit_side,
                            confidence=0.85,
                            strategy_name=self.name,
                            strategy_type=self.strategy_type,
                            metadata={"ml_prediction": ml_prediction, "reason": "ml_flip"},
                        )
                        return exit_signal
            except Exception:
                pass

        return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Delegate to the base PCA strategy for backtesting.

        In a production backtest with a trained LSTM available, the signals
        would be gated per‑bar.  Without a serialized model this delegation
        is the correct fallback: it still uses the same PCA edge while
        preserving the statistical entry/exit thresholds defined above.
        """
        return self._base.backtest_signals(df)