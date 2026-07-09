"""ml_pca_arb.py
ML‑Enhanced PCA Statistical Arbitrage Strategy.

This module defines :class:`MLPCAStatArbStrategy`, which extends the classic
PCA statistical‑arbitrage approach by gating entry signals through an LSTM
confidence filter.  The strategy works as follows:

1. The underlying :class:`~app.strategies.manual.pca_stat_arb.PCAStatArbStrategy`
   computes a PCA s‑score and emits a base :class:`~app.strategies.base.Signal`.
2. If an ML inference service is available, the LSTM model predicts a direction
   (``up``/``down``) and a confidence score.  The trade is taken only when the
   LSTM prediction agrees with the PCA side **and** the confidence exceeds a
   configurable threshold.
3. When the ML service cannot be imported or raises an exception, the strategy
   degrades gracefully – ``analyze`` returns ``None`` (no trade) and back‑testing
   falls back to the base PCA logic.

The implementation purposefully avoids any hard‑coded data or paid API calls,
and it preserves the original behaviour while adding comprehensive
documentation and type annotations.
"""

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, Optional

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.pca_stat_arb import PCAStatArbStrategy

# ----------------------------------------------------------------------
# Optional ML inference import – defensive loading.
# ----------------------------------------------------------------------
try:
    from app.ml.inference import get_inference_service as _get_inference_service
    _INFERENCE_AVAILABLE: bool = True
except Exception:
    _INFERENCE_AVAILABLE = False

# Default confidence threshold used when the caller does not provide one.
_ML_CONFIDENCE_THRESHOLD: float = 0.60


class MLPCAStatArbStrategy(AbstractStrategy):
    """
    ML‑gated PCA Statistical Arbitrage.

    The strategy mirrors :class:`PCAStatArbStrategy` but adds an LSTM filter.
    When the ML inference service is unavailable the strategy falls back
    gracefully:

    * ``analyze`` → returns ``None`` (no signal).
    * ``backtest_signals`` → delegates to the base PCA strategy.

    Attributes
    ----------
    name : str
        Internal identifier used by the framework.
    display_name : str
        Human‑readable name shown in UI / logs.
    market_type : str
        Market classification (e.g., ``"equity"``).
    strategy_type : str
        Category of the strategy – here ``"ml_enhanced"``.
    risk_bucket : str
        Risk classification for reporting.
    tick_interval_seconds : float
        Minimum interval between ticks; ``86_400`` seconds corresponds to daily.
    confidence_threshold : float
        Upper‑level confidence threshold used by the base class (not to be
        confused with the ML confidence threshold).
    _base : PCAStatArbStrategy
        Instance of the underlying PCA statistical‑arbitrage strategy.
    _ml_threshold : float
        Confidence threshold for the LSTM model; defaults to
        ``_ML_CONFIDENCE_THRESHOLD``.
    """

    name = "ml_pca_arb"
    display_name = "ML PCA Statistical Arbitrage (LSTM-Gated)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "arbitrage"
    tick_interval_seconds = 86_400.0  # daily
    confidence_threshold = 0.65

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the strategy.

        Parameters
        ----------
        params : dict | None, optional
            Optional configuration dictionary.  Recognised keys:

            * ``ml_confidence_threshold`` – float, overrides the default ML
              confidence gating threshold.
        """
        super().__init__(params)
        p = params or {}
        self._base: PCAStatArbStrategy = PCAStatArbStrategy(params)
        self._ml_threshold: float = float(
            p.get("ml_confidence_threshold", _ML_CONFIDENCE_THRESHOLD)
        )

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Produce a trade signal conditioned on both PCA and LSTM outputs.

        The method proceeds in three steps:

        1. Retrieve the base PCA signal via ``self._base.analyze``.
        2. If the ML inference service is available, obtain an LSTM prediction.
        3. Validate the LSTM confidence and direction against the PCA side.
           If all checks pass, blend the confidences and return the enriched
           :class:`Signal`; otherwise ``None`` is returned.

        Parameters
        ----------
        data : pd.DataFrame
            Market data required by both the PCA and LSTM components.
        symbol : str
            Ticker symbol for which the signal is being generated.

        Returns
        -------
        Signal | None
            An enriched signal when both models agree, otherwise ``None``.
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

            # Blend confidences (capped at 0.95)
            blended = min(0.95, (base_signal.confidence + ml_confidence) / 2)
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

        Delegates directly to the underlying :class:`PCAStatArbStrategy`.  In a
        production environment with a serialized LSTM model, this method could
        be extended to gate signals per bar; the current implementation provides
        a safe fallback.

        Parameters
        ----------
        df : pd.DataFrame
            Historical data used for back‑testing.

        Returns
        -------
        BacktestSignals
            Signals produced by the base PCA strategy.
        """
        return self._base.backtest_signals(df)