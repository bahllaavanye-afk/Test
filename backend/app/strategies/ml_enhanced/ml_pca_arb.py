"""
ML-Enhanced PCA Statistical Arbitrage Strategy.

This module defines :class:`MLPCAStatArbStrategy`, an extension of the
classic PCA statistical arbitrage approach that gates entry signals through
a lightweight LSTM confidence filter. A trade is taken only when both
conditions are satisfied:

1. The PCA s‑score exceeds the entry threshold (mean‑reversion signal).
2. The LSTM model predicts the same direction with confidence above the
   configured threshold.

If the optional ML inference service cannot be imported or raises an
exception, the strategy degrades gracefully – ``analyze`` returns ``None``
(no trade) and back‑testing falls back to the base PCA strategy.
"""

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, Optional

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.pca_stat_arb import PCAStatArbStrategy

# ----------------------------------------------------------------------
# Optional ML inference service
# ----------------------------------------------------------------------
try:
    from app.ml.inference import get_inference_service as _get_inference_service
    _INFERENCE_AVAILABLE: bool = True
except Exception:
    _INFERENCE_AVAILABLE: bool = False

_ML_CONFIDENCE_THRESHOLD: float = 0.60
"""Default confidence threshold used when the caller does not supply a custom value."""


class MLPCAStatArbStrategy(AbstractStrategy):
    """
    ML‑gated PCA Statistical Arbitrage.

    The strategy reuses the s‑score logic from :class:`PCAStatArbStrategy` but
    filters each entry signal through an LSTM model. When the ML service is
    unavailable the strategy degrades gracefully:

    * ``analyze`` → returns ``None`` (no signal)
    * ``backtest_signals`` → delegates to the base PCA strategy
    """

    name: str = "ml_pca_arb"
    display_name: str = "ML PCA Statistical Arbitrage (LSTM-Gated)"
    market_type: str = "equity"
    strategy_type: str = "ml_enhanced"
    risk_bucket: str = "arbitrage"
    tick_interval_seconds: float = 86_400.0  # daily
    confidence_threshold: float = 0.65

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the strategy.

        Parameters
        ----------
        params :
            Optional dictionary of configuration parameters. Recognised
            keys:

            * ``ml_confidence_threshold`` – float, overrides the default
              confidence threshold used to gate LSTM predictions.
        """
        super().__init__(params)
        p: Dict[str, Any] = params or {}
        self._base: PCAStatArbStrategy = PCAStatArbStrategy(params)
        self._ml_threshold: float = float(
            p.get("ml_confidence_threshold", _ML_CONFIDENCE_THRESHOLD)
        )

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Produce a trade signal when both PCA and LSTM agree.

        The method first obtains a base signal from :class:`PCAStatArbStrategy`.
        If the optional ML inference service is available, it queries the LSTM
        model for a confidence score and direction prediction. The signal is
        returned only when:

        * The LSTM confidence meets or exceeds ``self._ml_threshold``.
        * The LSTM prediction is not ``"neutral"``.
        * The prediction direction matches the PCA side (``"buy"``/``"sell"``).

        When the ML service is unavailable or any check fails, ``None`` is
        returned, indicating no trade.

        Parameters
        ----------
        data :
            DataFrame containing the required market data for the given
            ``symbol``.
        symbol :
            Ticker symbol for which the signal is being generated.

        Returns
        -------
        Signal | None
            A populated :class:`Signal` instance when conditions are met,
            otherwise ``None``.
        """
        # Step 1: get base PCA signal
        base_signal: Optional[Signal] = await self._base.analyze(data, symbol)
        if base_signal is None:
            return None

        # Step 2: apply ML filter
        if not _INFERENCE_AVAILABLE:
            # ML service not installed — skip silently
            return None

        try:
            inference = _get_inference_service()
            ml_result: Optional[Dict[str, Any]] = await inference.predict(data, symbol)
            if ml_result is None:
                return None

            ml_confidence: float = float(ml_result.get("confidence", 0.0))
            ml_prediction: str = ml_result.get("prediction", "neutral")

            if ml_confidence < self._ml_threshold or ml_prediction == "neutral":
                return None

            # Direction agreement check
            direction_ok: bool = (
                (ml_prediction == "up" and base_signal.side == "buy")
                or (ml_prediction == "down" and base_signal.side == "sell")
            )
            if not direction_ok:
                return None

            # Blend confidences
            blended: float = min(0.95, (base_signal.confidence + ml_confidence) / 2)
            base_signal.confidence = blended
            base_signal.strategy_name = self.name
            base_signal.strategy_type = self.strategy_type
            base_signal.metadata["ml_confidence"] = ml_confidence
            return base_signal

        except Exception:
            # ML service raised an error — degrade gracefully
            return None

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Generate back‑testing signals.

        For back‑testing the strategy simply delegates to the underlying
        :class:`PCAStatArbStrategy`. When a serialized LSTM model is available,
        a more sophisticated per‑bar gating could be implemented, but the
        fallback ensures consistent historical performance evaluation.

        Parameters
        ----------
        df :
            DataFrame containing historical market data.

        Returns
        -------
        BacktestSignals
            The signals produced by the base PCA strategy.
        """
        return self._base.backtest_signals(df)