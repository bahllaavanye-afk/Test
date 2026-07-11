"""
ML-Enhanced PCA Statistical Arbitrage Strategy.

Extends PCAStatArbStrategy by gating entries through an LSTM confidence
filter and additional market‑condition confirmations. A trade is only taken
when **all** conditions are true:

  1. PCA s‑score exceeds the entry threshold (mean‑reversion signal)
  2. LSTM model confidence > configurable threshold (directional agreement)
  3. Recent price volatility is below a configurable limit (to avoid noisy
     regimes)
  4. Recent volume is above a configurable percentile (to ensure liquidity)

If the ML inference service is unavailable the strategy falls back
gracefully (returns ``None`` from ``analyze``, uses base signals in back‑test).
"""

import logging
from typing import Any

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.pca_stat_arb import PCAStatArbStrategy

# ----------------------------------------------------------------------
# Optional ML inference import – keep the strategy functional without it
# ----------------------------------------------------------------------
try:
    from app.ml.inference import get_inference_service as _get_inference_service
    _INFERENCE_AVAILABLE = True
except Exception:  # pragma: no cover
    _INFERENCE_AVAILABLE = False

# ----------------------------------------------------------------------
# Default configuration constants
# ----------------------------------------------------------------------
_ML_CONFIDENCE_THRESHOLD = 0.60
_VOLATILITY_WINDOW = 5          # days
_MAX_VOLATILITY_PCT = 0.02      # 2 %
_VOLUME_WINDOW = 5              # days
_VOLUME_PERCENTILE = 0.25       # 25 % of historical volume

_logger = logging.getLogger(__name__)


class MLPCAStatArbStrategy(AbstractStrategy):
    """
    ML‑gated PCA Statistical Arbitrage with extra confirmation filters.

    The core s‑score logic is delegated to :class:`PCAStatArbStrategy`.  This
    wrapper adds:
      * LSTM confidence gating
      * Volatility and volume confirmation filters
      * A more nuanced confidence blending scheme
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

        # Configurable thresholds – fall back to module defaults
        self._ml_threshold: float = float(
            p.get("ml_confidence_threshold", _ML_CONFIDENCE_THRESHOLD)
        )
        self._max_volatility: float = float(
            p.get("max_volatility_pct", _MAX_VOLATILITY_PCT)
        )
        self._volume_percentile: float = float(
            p.get("volume_percentile", _VOLOLUME_PERCENTILE)
        )

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _calc_volatility(series: pd.Series) -> float:
        """Return the absolute % change over the configured window."""
        if series.empty:
            return 0.0
        return abs(series.pct_change().dropna()).mean()

    @staticmethod
    def _calc_volume_percentile(volume_series: pd.Series, window: int) -> float:
        """Compute the percentile rank of the latest volume within the window."""
        if volume_series.empty:
            return 0.0
        recent = volume_series.tail(window)
        latest = recent.iloc[-1]
        rank = (recent < latest).sum() / len(recent)
        return rank

    def _passes_additional_filters(self, data: pd.DataFrame) -> bool:
        """
        Apply volatility and volume confirmations.

        Returns ``True`` if the market conditions are deemed suitable for a
        PCA‑based entry.
        """
        # Expect columns 'close' and 'volume' – fail safe if missing
        if not {"close", "volume"}.issubset(data.columns):
            _logger.debug("Data missing required columns for additional filters.")
            return False

        # Volatility filter
        volatility = self._calc_volatility(data["close"])
        if volatility > self._max_volatility:
            _logger.debug(
                "Volatility %.4f exceeds max allowed %.4f",
                volatility,
                self._max_volatility,
            )
            return False

        # Volume filter – ensure recent volume is in the top percentile
        vol_percentile = self._calc_volume_percentile(data["volume"], _VOLUME_WINDOW)
        if vol_percentile < self._volume_percentile:
            _logger.debug(
                "Volume percentile %.2f below required %.2f",
                vol_percentile,
                self._volume_percentile,
            )
            return False

        return True

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------
    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Generate a signal only when PCA s‑score, LSTM confidence, and market
        confirmations all agree.

        If the ML service is unavailable or any filter fails, ``None`` is
        returned, signalling no trade.
        """
        # 1️⃣ Base PCA signal
        base_signal = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # 2️⃣ Market‑condition confirmations
        if not self._passes_additional_filters(data):
            return None

        # 3️⃣ LSTM confidence gating
        if not _INFERENCE_AVAILABLE:
            _logger.debug("ML inference service unavailable – skipping ML filter.")
            return None

        try:
            inference = _get_inference_service()
            ml_result: dict[str, Any] | None = await inference.predict(data, symbol)
            if not ml_result:
                return None

            ml_confidence = float(ml_result.get("confidence", 0.0))
            ml_prediction = ml_result.get("prediction", "neutral")

            if ml_confidence < self._ml_threshold:
                _logger.debug(
                    "ML confidence %.3f below threshold %.3f",
                    ml_confidence,
                    self._ml_threshold,
                )
                return None

            if ml_prediction == "neutral":
                _logger.debug("ML prediction neutral – rejecting entry.")
                return None

            # 4️⃣ Direction agreement
            direction_ok = (
                (ml_prediction == "up" and base_signal.side == "buy")
                or (ml_prediction == "down" and base_signal.side == "sell")
            )
            if not direction_ok:
                _logger.debug(
                    "ML prediction %s disagrees with base side %s",
                    ml_prediction,
                    base_signal.side,
                )
                return None

            # 5️⃣ Confidence blending – weight ML higher when its confidence
            #    exceeds the base confidence
            weight_ml = 0.6 if ml_confidence > base_signal.confidence else 0.4
            blended_confidence = min(
                0.99,
                weight_ml * ml_confidence + (1 - weight_ml) * base_signal.confidence,
            )
            base_signal.confidence = blended_confidence
            base_signal.strategy_name = self.name
            base_signal.strategy_type = self.strategy_type
            base_signal.metadata.update(
                {
                    "ml_confidence": ml_confidence,
                    "ml_prediction": ml_prediction,
                    "volatility": volatility,
                    "volume_percentile": vol_percentile,
                }
            )
            return base_signal

        except Exception as exc:  # pragma: no cover
            _logger.exception("Error during ML inference: %s", exc)
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Delegates to the base PCA strategy for back‑testing.

        When a serialized LSTM model is available, the back‑test runner can
        replace this method with a per‑bar gated version.  The fallback keeps
        the strategy functional in all environments.
        """
        return self._base.backtest_signals(df)