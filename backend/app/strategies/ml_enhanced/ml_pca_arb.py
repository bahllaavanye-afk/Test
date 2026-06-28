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

import logging
import time
from typing import Any, Dict, Optional

import pandas as pd

from app.strategies.base import AbstractStrategy, BacktestSignals, Signal
from app.strategies.manual.pca_stat_arb import PCAStatArbStrategy

# ML inference is optional — import defensively
try:
    from app.ml.inference import get_inference_service as _get_inference_service

    _INFERENCE_AVAILABLE = True
except Exception:
    _INFERENCE_AVAILABLE = False

_ML_CONFIDENCE_THRESHOLD: float = 0.60

_logger = logging.getLogger(__name__)


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
        # Monitoring counters
        self._signal_count: int = 0

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
            An enriched :class:`Signal` instance when both models agree,
            otherwise ``None``.
        """
        start_time = time.perf_counter()

        # Edge‑case guard: ensure inputs are valid
        if not symbol:
            _logger.warning(
                "MLPCAStatArbStrategy - received empty symbol",
                extra={"execution_time_ms": (time.perf_counter() - start_time) * 1000},
            )
            return None

        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            _logger.warning(
                "MLPCAStatArbStrategy - received invalid or empty data",
                extra={"symbol": symbol, "execution_time_ms": (time.perf_counter() - start_time) * 1000},
            )
            return None

        # Step 1: get base PCA signal
        base_signal: Optional[Signal] = await self._base.analyze(data, symbol)
        if base_signal is None:
            exec_ms = (time.perf_counter() - start_time) * 1000
            _logger.info(
                "MLPCAStatArbStrategy - no base PCA signal",
                extra={"symbol": symbol, "execution_time_ms": exec_ms},
            )
            return None

        # Defensive check: ensure the base signal has the required attributes
        if not hasattr(base_signal, "side") or not hasattr(base_signal, "confidence"):
            exec_ms = (time.perf_counter() - start_time) * 1000
            _logger.error(
                "MLPCAStatArbStrategy - base signal missing required attributes",
                extra={"symbol": symbol, "execution_time_ms": exec_ms},
            )
            return None

        # Step 2: apply ML filter
        if not _INFERENCE_AVAILABLE:
            exec_ms = (time.perf_counter() - start_time) * 1000
            _logger.info(
                "MLPCAStatArbStrategy - ML inference unavailable, skipping signal",
                extra={"symbol": symbol, "execution_time_ms": exec_ms},
            )
            return None

        try:
            inference = _get_inference_service()
            ml_result: Optional[Dict[str, Any]] = await inference.predict(data, symbol)

            if not ml_result:
                exec_ms = (time.perf_counter() - start_time) * 1000
                _logger.info(
                    "MLPCAStatArbStrategy - ML prediction returned None or empty",
                    extra={"symbol": symbol, "execution_time_ms": exec_ms},
                )
                return None

            # Extract and validate confidence
            raw_confidence = ml_result.get("confidence")
            try:
                ml_confidence: float = float(raw_confidence) if raw_confidence is not None else 0.0
            except (TypeError, ValueError):
                ml_confidence = 0.0

            # Extract prediction safely
            ml_prediction: str = str(ml_result.get("prediction", "neutral")).lower()

            if ml_confidence < self._ml_threshold:
                exec_ms = (time.perf_counter() - start_time) * 1000
                _logger.info(
                    "MLPCAStatArbStrategy - ML confidence below threshold",
                    extra={
                        "symbol": symbol,
                        "ml_confidence": ml_confidence,
                        "threshold": self._ml_threshold,
                        "execution_time_ms": exec_ms,
                    },
                )
                return None

            if ml_prediction == "neutral":
                exec_ms = (time.perf_counter() - start_time) * 1000
                _logger.info(
                    "MLPCAStatArbStrategy - ML prediction neutral",
                    extra={"symbol": symbol, "execution_time_ms": exec_ms},
                )
                return None

            # Direction agreement check
            direction_ok: bool = (
                (ml_prediction == "up" and base_signal.side == "buy")
                or (ml_prediction == "down" and base_signal.side == "sell")
            )
            if not direction_ok:
                exec_ms = (time.perf_counter() - start_time) * 1000
                _logger.info(
                    "MLPCAStatArbStrategy - direction mismatch between ML and PCA",
                    extra={
                        "symbol": symbol,
                        "ml_prediction": ml_prediction,
                        "pca_side": base_signal.side,
                        "execution_time_ms": exec_ms,
                    },
                )
                return None

            # Blend confidences (capped at 0.95) – guard against off‑by‑one rounding
            blended: float = min(0.95, (base_signal.confidence + ml_confidence) / 2.0)
            base_signal.confidence = blended
            base_signal.strategy_name = self.name
            base_signal.strategy_type = self.strategy_type
            # Ensure metadata dict exists
            if not isinstance(base_signal.metadata, dict):
                base_signal.metadata = {}
            base_signal.metadata["ml_confidence"] = ml_confidence

            # Monitoring: update counters and log key metrics
            self._signal_count += 1
            exec_ms = (time.perf_counter() - start_time) * 1000
            _logger.info(
                "MLPCAStatArbStrategy - signal generated",
                extra={
                    "symbol": symbol,
                    "signal_count": self._signal_count,
                    "execution_time_ms": exec_ms,
                    "side": base_signal.side,
                    "confidence": base_signal.confidence,
                    "ml_confidence": ml_confidence,
                    "pnl": getattr(base_signal, "pnl", None),
                },
            )
            return base_signal

        except Exception as exc:
            exec_ms = (time.perf_counter() - start_time) * 1000
            _logger.exception(
                "MLPCAStatArbStrategy - unexpected error during ML filtering",
                extra={"symbol": symbol, "execution_time_ms": exec_ms, "error": str(exc)},
            )
            return None

    async def backtest_signals(self, data: pd.DataFrame, symbol: str) -> BacktestSignals:
        """
        Generate backtest signals.

        Delegates directly to the underlying PCA strategy, ensuring that
        backtesting does not depend on the ML service (which may be unavailable
        or non‑deterministic in a historical context).

        Parameters
        ----------
        data : pandas.DataFrame
            Historical market data.
        symbol : str
            Ticker symbol.

        Returns
        -------
        BacktestSignals
            Signals produced by the base PCA strategy.
        """
        # Edge‑case guard: identical to ``analyze`` – early exit for invalid inputs
        if not symbol or data is None or not isinstance(data, pd.DataFrame) or data.empty:
            _logger.warning(
                "MLPCAStatArbStrategy - backtest called with invalid inputs, returning empty signals",
                extra={"symbol": symbol},
            )
            return BacktestSignals(signals=[])

        return await self._base.backtest_signals(data, symbol)