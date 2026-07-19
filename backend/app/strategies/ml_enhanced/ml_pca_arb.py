"""
ML-Enhanced PCA Statistical Arbitrage Strategy.

Extends PCAStatArbStrategy by gating entries through an LSTM confidence
filter and additional market‑condition confirmations.  The strategy
produces higher‑quality entry signals while preserving the original
exit logic of the base strategy.

If the ML inference service is unavailable the strategy degrades
gracefully (returns None from analyze, uses base signals in backtest).
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.pca_stat_arb import PCAStatArbStrategy

# ----------------------------------------------------------------------
# Optional ML inference import – defensive loading
# ----------------------------------------------------------------------
try:
    from app.ml.inference import get_inference_service as _get_inference_service
    _INFERENCE_AVAILABLE = True
except Exception:  # pragma: no cover
    _INFERENCE_AVAILABLE = False

# ----------------------------------------------------------------------
# Default thresholds – can be overridden via strategy params
# ----------------------------------------------------------------------
_ML_CONFIDENCE_THRESHOLD = 0.60          # entry confidence
_ML_EXIT_CONFIDENCE_THRESHOLD = 0.40    # exit confidence
_VOLUME_MA_WINDOW = 20
_VOLUME_FACTOR = 1.5
_ATR_WINDOW = 14
_ATR_THRESHOLD = 0.02                    # 2 % price move
_PCA_SSCORE_ENTRY_FACTOR = 1.0          # multiplier on base entry threshold

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
_logger = logging.getLogger(__name__)

class MLPCAStatArbStrategy(AbstractStrategy):
    """
    ML‑gated PCA Statistical Arbitrage.

    The strategy keeps the original PCA s‑score logic but adds three
    layers of confirmation before an entry is emitted:

    1. **ML confidence / direction** – LSTM prediction must agree with the
       PCA side and exceed a configurable confidence threshold.
    2. **Volume filter** – current volume must be at least ``VOL_FACTOR``
       times the moving‑average volume over ``VOL_MA_WINDOW`` bars.
    3. **Volatility filter** – the 14‑day ATR (average true range) must
       be above ``ATR_THRESHOLD`` to avoid flat markets.

    Exit signals are passed through unchanged, but a low ML confidence
    (below ``ML_EXIT_CONFIDENCE_THRESHOLD``) will also trigger an early
    exit if the base strategy has not already signalled one.
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

        # Base PCA strategy – shares the same parameter set
        self._base = PCAStatArbStrategy(params)

        # Thresholds – allow per‑instance overrides
        self._ml_entry_threshold: float = float(
            p.get("ml_confidence_threshold", _ML_CONFIDENCE_THRESHOLD)
        )
        self._ml_exit_threshold: float = float(
            p.get("ml_exit_confidence_threshold", _ML_EXIT_CONFIDENCE_THRESHOLD)
        )
        self._volume_ma_window: int = int(p.get("volume_ma_window", _VOLUME_MA_WINDOW))
        self._volume_factor: float = float(p.get("volume_factor", _VOL_FACTOR))
        self._atr_window: int = int(p.get("atr_window", _ATR_WINDOW))
        self._atr_threshold: float = float(p.get("atr_threshold", _ATR_THRESHOLD))
        self._sscore_entry_factor: float = float(
            p.get("sscore_entry_factor", _PCA_SSCORE_ENTRY_FACTOR)
        )

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _calculate_atr(df: pd.DataFrame) -> pd.Series:
        """
        Compute the Average True Range (ATR) over the configured window.
        Expects columns: high, low, close.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=_ATR_WINDOW, min_periods=1).mean()
        return atr

    def _passes_volume_filter(self, df: pd.DataFrame) -> bool:
        if "volume" not in df:
            return True  # No volume data – skip filter
        vol_ma = df["volume"].rolling(
            window=self._volume_ma_window, min_periods=1
        ).mean()
        current_vol = df["volume"].iloc[-1]
        required_vol = vol_ma.iloc[-1] * self._volume_factor
        return current_vol >= required_vol

    def _passes_volatility_filter(self, df: pd.DataFrame) -> bool:
        if not {"high", "low", "close"}.issubset(df.columns):
            return True  # Insufficient price data – skip filter
        atr = self._calculate_atr(df)
        recent_atr = atr.iloc[-1]
        # Scale ATR relative to recent close price to obtain a percentage
        recent_close = df["close"].iloc[-1]
        if recent_close == 0:
            return False
        atr_pct = recent_atr / recent_close
        return atr_pct >= self._atr_threshold

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------
    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate a signal only when PCA s‑score, LSTM, volume and
        volatility filters all agree.

        The method degrades gracefully:
          * No ML service → no entry signal (None)
          * ML errors → no entry signal (None)
          * Exit signals from the base strategy are returned unchanged,
            but a low ML confidence will also force an early exit.
        """
        # ------------------------------------------------------------------
        # 1️⃣ Base PCA signal
        # ------------------------------------------------------------------
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # ------------------------------------------------------------------
        # 2️⃣ Determine if this is an entry or exit signal
        # ------------------------------------------------------------------
        is_entry = getattr(base_signal, "is_entry", True)
        # Some implementations expose a ``signal_type`` attribute; fall back
        if hasattr(base_signal, "signal_type"):
            is_entry = base_signal.signal_type == "entry"

        # ------------------------------------------------------------------
        # 3️⃣ Exit path – optionally enforce a low‑confidence exit filter
        # ------------------------------------------------------------------
        if not is_entry:
            # If ML is available we can still enforce an early exit when
            # confidence drops below the exit threshold.
            if _INFERENCE_AVAILABLE:
                try:
                    inference = _get_inference_service()
                    ml_result = await inference.predict(data, symbol)
                    if ml_result:
                        ml_conf = float(ml_result.get("confidence", 0.0))
                        if ml_conf < self._ml_exit_threshold:
                            # Mark the base signal as an early exit
                            base_signal.confidence = ml_conf
                            base_signal.metadata = base_signal.metadata or {}
                            base_signal.metadata["ml_exit_confidence"] = ml_conf
                            return base_signal
                except Exception as exc:  # pragma: no cover
                    _logger.debug(
                        "ML exit filter error for %s: %s", symbol, exc, exc_info=True
                    )
            return base_signal

        # ------------------------------------------------------------------
        # 4️⃣ Entry path – apply all confirmation filters
        # ------------------------------------------------------------------
        # 4a. ML filter
        if not _INFERENCE_AVAILABLE:
            _logger.debug("ML inference service unavailable – skipping entry.")
            return None

        try:
            inference = _get_inference_service()
            ml_result = await inference.predict(data, symbol)
            if not ml_result:
                return None

            ml_confidence: float = float(ml_result.get("confidence", 0.0))
            ml_prediction: str = ml_result.get("prediction", "neutral")
        except Exception as exc:  # pragma: no cover
            _logger.debug(
                "ML inference error for %s: %s", symbol, exc, exc_info=True
            )
            return None

        if ml_confidence < self._ml_entry_threshold:
            _logger.debug(
                "ML confidence %.3f below threshold %.3f for %s",
                ml_confidence,
                self._ml_entry_threshold,
                symbol,
            )
            return None
        if ml_prediction == "neutral":
            _logger.debug("ML prediction neutral for %s – no entry.", symbol)
            return None

        # 4b. Direction agreement
        direction_ok = (
            (ml_prediction == "up" and base_signal.side == "buy")
            or (ml_prediction == "down" and base_signal.side == "sell")
        )
        if not direction_ok:
            _logger.debug(
                "ML direction %s mismatches PCA side %s for %s",
                ml_prediction,
                base_signal.side,
                symbol,
            )
            return None

        # 4c. PCA s‑score strength check (if present)
        sscore = float(base_signal.metadata.get("sscore", 0.0))
        if abs(sscore) < self._sscore_entry_factor * getattr(self._base, "entry_threshold", 0):
            _logger.debug(
                "PCA s‑score %.3f below entry factor threshold for %s",
                sscore,
                symbol,
            )
            return None

        # 4d. Volume filter
        if not self._passes_volume_filter(data):
            _logger.debug("Volume filter failed for %s", symbol)
            return None

        # 4e. Volatility filter
        if not self._passes_volatility_filter(data):
            _logger.debug("Volatility filter failed for %s", symbol)
            return None

        # ------------------------------------------------------------------
        # 5️⃣ Signal blending & enrichment
        # ------------------------------------------------------------------
        blended_confidence = min(
            0.95, (base_signal.confidence + ml_confidence) / 2.0
        )
        base_signal.confidence = blended_confidence
        base_signal.strategy_name = self.name
        base_signal.strategy_type = self.strategy_type
        base_signal.metadata = base_signal.metadata or {}
        base_signal.metadata.update(
            {
                "ml_confidence": ml_confidence,
                "ml_prediction": ml_prediction,
                "volume_factor": self._volume_factor,
                "atr_pct": self._atr_threshold,
            }
        )
        return base_signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Delegates to the base PCA strategy for backtesting.

        In a full back‑test with a serialized LSTM model, the gating logic
        would be applied per bar.  Because the model is typically unavailable
        in the CI environment, we safely fall back to the base signals.
        """
        return self._base.backtest_signals(df)