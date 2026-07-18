"""
ML-Enhanced PCA Statistical Arbitrage Strategy.

Extends PCAStatArbStrategy by gating entries through an LSTM confidence
filter and adding extra market‑condition confirmations. Exit logic is
tightened by monitoring the PCA s‑score reversion and basic volatility
filters.

If the ML inference service is unavailable the strategy falls back
gracefully (returns None from analyze, uses base signals in backtest).
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
# Minimum price volatility (5‑day pct‑change std) required for a trade
_MIN_VOLATILITY = 0.001
# Minimum average daily volume required for a trade (placeholder)
_MIN_VOLUME = 1_000_000


class MLPCAStatArbStrategy(AbstractStrategy):
    """
    ML‑gated PCA Statistical Arbitrage.

    Same s‑score logic as PCAStatArbStrategy but each entry signal is
    filtered through an LSTM model and additional market‑condition checks.
    When the ML service is not loaded the strategy degrades gracefully:
      - analyze()           → returns None (no signal)
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
        # Optional overrides for extra filters
        self._min_volatility: float = float(
            p.get("min_volatility", _MIN_VOLATILITY)
        )
        self._min_volume: float = float(
            p.get("min_volume", _MIN_VOLUME)
        )

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate a signal only when PCA s‑score, LSTM agree,
        and basic market‑condition filters pass.

        Falls back to None (no trade) when ML is unavailable or any filter
        fails.
        """
        # 1️⃣ Base PCA entry/exit signal
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # 2️⃣ Basic market‑condition confirmations
        if not self._passes_market_filters(data):
            return None

        # 3️⃣ ML gating – skip if inference service unavailable
        if not _INFERENCE_AVAILABLE:
            return None

        try:
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

            # 4️⃣ Blend confidences (cap to avoid over‑confidence)
            blended = min(0.95, (base_signal.confidence + ml_confidence) / 2)
            base_signal.confidence = blended
            base_signal.strategy_name = self.name
            base_signal.strategy_type = self.strategy_type
            base_signal.metadata["ml_confidence"] = ml_confidence

            # 5️⃣ Tightened exit handling – add exit trigger metadata
            exit_info = self._evaluate_exit_condition(data, base_signal)
            if exit_info:
                base_signal.metadata.update(exit_info)

            return base_signal

        except Exception:
            # ML service raised an error — degrade gracefully
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Delegate to the base PCA strategy for backtesting.

        In a production backtest with a trained LSTM available, the signals
        would be gated per‑bar.  Without a serialized model this delegation
        is the correct fallback: it still uses the same PCA edge.
        """
        return self._base.backtest_signals(df)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _passes_market_filters(self, data: pd.DataFrame) -> bool:
        """
        Apply lightweight market‑condition filters:

        * Minimum price volatility (5‑day rolling std of pct‑change)
        * Minimum average daily volume
        """
        # Volatility check
        if "price" in data.columns:
            price = data["price"]
            vol = price.pct_change().rolling(5).std().iloc[-1]
            if pd.isna(vol) or vol < self._min_volatility:
                return False
        else:
            return False

        # Volume check
        if "volume" in data.columns:
            avg_vol = data["volume"].rolling(5).mean().iloc[-1]
            if pd.isna(avg_vol) or avg_vol < self._min_volume:
                return False
        else:
            return False

        return True

    def _evaluate_exit_condition(self, data: pd.DataFrame, base_signal: Signal) -> dict | None:
        """
        Determine whether an exit condition is met based on the PCA s‑score
        reverting toward zero. Returns a dict with exit metadata if the
        condition is satisfied, otherwise None.
        """
        s_score = None
        if "s_score" in data.columns:
            s_score = data["s_score"].iloc[-1]

        # If the strategy provides explicit thresholds, use them
        entry_thr = getattr(self._base, "entry_threshold", None)
        exit_thr = getattr(self._base, "exit_threshold", None)

        if s_score is None:
            return None

        # Exit when the absolute s‑score falls below the exit threshold
        if exit_thr is not None and abs(s_score) < exit_thr:
            return {
                "exit_trigger": "s_score_reversion",
                "s_score": float(s_score),
                "exit_threshold": float(exit_thr),
            }

        # Fallback: tighten exit when s‑score crosses half the entry threshold
        if entry_thr is not None and abs(s_score) < (entry_thr / 2):
            return {
                "exit_trigger": "s_score_half_entry",
                "s_score": float(s_score),
                "entry_threshold": float(entry_thr),
            }

        return None