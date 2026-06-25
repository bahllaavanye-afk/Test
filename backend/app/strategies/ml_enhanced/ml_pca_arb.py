"""
ML-Enhanced PCA Statistical Arbitrage Strategy.

Extends :class:`~app.strategies.manual.pca_stat_arb.PCAStatArbStrategy` by
gating entries through an LSTM confidence filter: a trade is only taken
when **both** conditions are true:

1. PCA s‑score exceeds the entry threshold (mean‑reversion signal)
2. LSTM model confidence > 0.60 (directional agreement)

If the ML inference service is unavailable the strategy falls back
gracefully (returns ``None`` from :meth:`analyze`, uses base signals in
backtest).
"""

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, Optional

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.pca_stat_arb import PCAStatArbStrategy

# ML inference is optional — import defensively
try:
    from app.ml.inference import get_inference_service as _get_inference_service
    _INFERENCE_AVAILABLE = True
except Exception:
    _INFERENCE_AVAILABLE = False


_ML_CONFIDENCE_THRESHOLD: float = 0.60


class MLPCAStatArbStrategy(AbstractStrategy):
    """
    ML‑gated PCA Statistical Arbitrage.

    The core s‑score logic is delegated to :class:`PCAStatArbStrategy`.  When
    an LSTM model is available, each entry signal is filtered through the
    model; the trade proceeds only if the model’s confidence exceeds the
    configured threshold and its directional prediction agrees with the
    PCA signal.

    Graceful degradation:

    * ``analyze`` → returns ``None`` (no signal) when the ML service cannot
      be reached.
    * ``backtest_signals`` → delegates to the base PCA strategy.
    """

    name: str = "ml_pca_arb"
    display_name: str = "ML PCA Statistical Arbitrage (LSTM-Gated)"
    market_type: str = "equity"
    strategy_type: str = "ml_enhanced"
    risk_bucket: str = "arbitrage"
    tick_interval_seconds: float = 86_400.0  # daily
    confidence_threshold: float = 0.65

    def __init__(self, params: Optional[dict] = None) -> None:
        """
        Initialise the strategy.

        Parameters
        ----------
        params : dict | None, optional
            Optional configuration dictionary. Recognised keys:

            * ``ml_confidence_threshold`` – float, confidence threshold for the
              LSTM model. Defaults to :data:`_ML_CONFIDENCE_THRESHOLD`.
        """
        super().__init__(params)
        p: dict = params or {}
        self._base: PCAStatArbStrategy = PCAStatArbStrategy(params)
        self._ml_threshold: float = float(
            p.get("ml_confidence_threshold", _ML_CONFIDENCE_THRESHOLD)
        )

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Produce a trade signal when both the PCA and LSTM models agree.

        The method first obtains a base signal from the underlying
        :class:`PCAStatArbStrategy`.  If the ML inference service is available,
        the LSTM model is queried; its confidence and directional prediction are
        used to filter the base signal.  When any step fails or the
        confidence/direction criteria are not met, ``None`` is returned.

        Parameters
        ----------
        data : pandas.DataFrame
            Market data required by both the PCA and LSTM models.
        symbol : str
            Ticker symbol for which the signal is being generated.

        Returns
        -------
        Signal | None
            A enriched :class:`Signal` instance when both models agree,
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

            if ml_confidence < self._ml_threshold:
                return None
            if ml_prediction == "neutral":
                return None

            # Direction agreement check
            direction_ok: bool = (
                (ml_prediction == "up" and base_signal.side == "buy")
                or (ml_prediction == "down" and base_signal.side == "sell")
            )
            if not direction_ok:
                return None

            # Blend confidences (capped at 0.95)
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
        Generate back‑test signals.

        The back‑test implementation simply forwards the request to the base
        PCA strategy.  When an LSTM model is available, a production back‑test
        would gate signals per bar, but that logic is outside the scope of this
        fallback implementation.

        Parameters
        ----------
        df : pandas.DataFrame
            Historical price data for back‑testing.

        Returns
        -------
        BacktestSignals
            The signal set produced by the underlying PCA strategy.
        """
        return self._base.backtest_signals(df)