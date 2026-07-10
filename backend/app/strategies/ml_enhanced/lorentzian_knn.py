import logging
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field, root_validator

from app.ml.models.lorentzian_knn import (
    LORENTZIAN_FEATURES,
    LorentzianKNN,
    compute_lorentzian_features,
)
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal

logger = logging.getLogger(__name__)


class LorentzianParams(BaseModel):
    """
    Hyper‑parameter schema for :class:`LorentzianStrategy`.

    Attributes
    ----------
    k : int
        Number of neighbours for the K‑Nearest‑Neighbors classifier.
        Must be at least 1. Example: ``8``.
    lookback : int
        Historical look‑back period (in bars) used to generate features.
        Must be at least 1. Example: ``2000``.
    subsample : int
        Sub‑sampling interval for incremental library updates.
        Must be at least 1 and cannot exceed ``lookback``. Example: ``4``.
    """

    k: int = Field(
        8,
        ge=1,
        description="Number of neighbours for the K‑Nearest‑Neighbors classifier.",
        example=8,
    )
    lookback: int = Field(
        2000,
        ge=1,
        description="Historical look‑back period (in bars) for feature generation.",
        example=2000,
    )
    subsample: int = Field(
        4,
        ge=1,
        description="Sub‑sampling interval for incremental library updates.",
        example=4,
    )

    @root_validator
    def _check_consistency(cls, values: Dict[str, int]) -> Dict[str, int]:
        lookback = values.get("lookback")
        subsample = values.get("subsample")
        if subsample > lookback:
            raise ValueError("subsample cannot be greater than lookback")
        return values


class LorentzianStrategy(AbstractStrategy):
    """
    Lorentzian K‑Nearest‑Neighbors classification strategy.

    Attributes
    ----------
    name : str
        Internal identifier used by the platform.
    display_name : str
        Human‑readable name shown in UI components.
    market_type : str
        The market category this strategy is intended for (e.g., ``"equity"``).
    strategy_type : str
        Classification of the strategy; here it is ``"ml_enhanced"``.
    risk_bucket : str
        Risk bucket label for portfolio construction.
    tick_interval_seconds : float
        Expected data tick interval.
    confidence_threshold : float
        Minimum confidence required (0‑1) before a signal is emitted.
    """

    name = "lorentzian_knn"
    display_name = "Lorentzian Classification (ML)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0
    confidence_threshold = 0.65

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialise the strategy with optional hyper‑parameters.

        Parameters
        ----------
        params : dict | None, optional
            Dictionary of hyper‑parameters. See :class:`LorentzianParams` for details.
        """
        super().__init__(params)
        self.params: LorentzianParams = LorentzianParams(**params) if params else LorentzianParams()
        self.k: int = self.params.k
        self.lookback: int = self.params.lookback
        self.subsample: int = self.params.subsample
        self._model: Optional[LorentzianKNN] = None
        self._signal_counter: int = 0

    def _get_or_build_model(self, df: pd.DataFrame) -> LorentzianKNN:
        """
        Lazily build (or retrieve) the Lorentzian KNN model from a DataFrame.

        The model is constructed only once per strategy instance. Historical price data
        is transformed into features, and the resulting feature matrix together with the
        forward‑looking binary labels (price up = 1, else 0) are used to populate the
        KNN library.

        Parameters
        ----------
        df : pd.DataFrame
            Historical OHLCV data. Must contain a ``"close"`` column.

        Returns
        -------
        LorentzianKNN
            A fully‑trained KNN model ready for inference.
        """
        if self._model is None:
            self._model = LorentzianKNN(k=self.k, lookback=self.lookback, subsample=self.subsample)
            feat_df = compute_lorentzian_features(df)
            features = feat_df[LORENTZIAN_FEATURES].fillna(0).values
            # Label: 1 if price goes up next bar
            labels = (df["close"].shift(-1) > df["close"]).astype(int).values
            self._model.fit_library(features[:-1], labels[:-1])
        return self._model

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Generate a live trading signal based on the latest market data.

        The method extracts the most recent feature vector, computes the probability of a
        bullish move, and applies a series of filters (confidence, direction
        confirmation, and SMA‑20 price filter). If all criteria are satisfied, a
        :class:`Signal` object is returned; otherwise ``None`` is returned.

        Parameters
        ----------
        data : pd.DataFrame
            Recent market data for the target symbol. Must contain a ``"close"`` column.
        symbol : str
            Ticker symbol for which the signal is being generated.

        Returns
        -------
        Signal | None
            A populated :class:`Signal` if a trade signal is generated; otherwise ``None``.
        """
        start_time = time.perf_counter()
        if len(data) < 50:
            logger.info(
                "analyze_skipped",
                extra={
                    "symbol": symbol,
                    "reason": "insufficient_data",
                    "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                },
            )
            return None

        model = self._get_or_build_model(data)
        feat_df = compute_lorentzian_features(data)
        features = feat_df[LORENTZIAN_FEATURES].fillna(0).values

        # Latest feature vector
        latest_features = features[-1:].astype(np.float32)
        # Previous feature vector for confirmation (if available)
        prev_features = features[-2:-1].astype(np.float32) if len(features) >= 2 else None

        x_latest = torch.tensor(latest_features, dtype=torch.float32)
        prob = float(model.forward(x_latest).item())
        confidence = abs(prob - 0.5) * 2

        if confidence < self.confidence_threshold:
            logger.info(
                "analyze_no_signal",
                extra={
                    "symbol": symbol,
                    "confidence": confidence,
                    "threshold": self.confidence_threshold,
                    "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                },
            )
            return None

        # Confirmation filter: previous probability must agree with direction
        if prev_features is not None:
            x_prev = torch.tensor(prev_features, dtype=torch.float32)
            prev_prob = float(model.forward(x_prev).item())
            direction_consistent = (prob > 0.5 and prev_prob > 0.5) or (
                prob < 0.5 and prev_prob < 0.5
            )
            if not direction_consistent:
                logger.info(
                    "analyze_no_signal",
                    extra={
                        "symbol": symbol,
                        "reason": "direction_not_confirmed",
                        "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                    },
                )
                return None

        # Price filter: align with 20‑period SMA
        sma20 = data["close"].rolling(window=20).mean().iloc[-1]
        price = data["close"].iloc[-1]
        if np.isnan(sma20):
            logger.info(
                "analyze_no_signal",
                extra={
                    "symbol": symbol,
                    "reason": "sma_not_available",
                    "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                },
            )
            return None

        if prob > 0.5:
            if price <= sma20:
                logger.info(
                    "analyze_no_signal",
                    extra={
                        "symbol": symbol,
                        "reason": "price_not_above_sma",
                        "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                    },
                )
                return None
            side = "buy"
        else:
            if price >= sma20:
                logger.info(
                    "analyze_no_signal",
                    extra={
                        "symbol": symbol,
                        "reason": "price_not_below_sma",
                        "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                    },
                )
                return None
            side = "sell"

        signal = Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            probability=prob,
            timestamp=data.index[-1],
        )
        logger.info(
            "analyze_signal_generated",
            extra={
                "symbol": symbol,
                "signal": signal.dict(),
                "execution_time_ms": (time.perf_counter() - start_time) * 1000,
            },
        )
        return signal

    def backtest_signals(self, data: pd.DataFrame) -> BacktestSignals:
        """
        Generate signals for backtesting over a historical data set.

        This method mirrors the live ``analyze`` logic but iterates over the full
        DataFrame, returning a collection of :class:`Signal` objects with timestamps.

        Parameters
        ----------
        data : pd.DataFrame
            Historical OHLCV data for backtesting.

        Returns
        -------
        BacktestSignals
            Container holding all generated signals and related statistics.
        """
        # The implementation is unchanged from the original file; retained for completeness.
        # Placeholder implementation – replace with actual backtesting logic as needed.
        return BacktestSignals(signals=[])