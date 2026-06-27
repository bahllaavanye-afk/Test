"""
Lorentzian KNN strategy — Python port of TradingView's most popular ML indicator.
Uses Lorentzian distance (robust to outliers) for k-nearest-neighbors classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from app.ml.models.lorentzian_knn import (
    LORENTZIAN_FEATURES,
    LorentzianKNN,
    compute_lorentzian_features,
)
from app.strategies.base import AbstractStrategy, BacktestSignals, Signal


class LorentzianStrategy(AbstractStrategy):
    """
    Lorentzian K-Nearest Neighbors strategy.

    This strategy builds a Lorentzian distance based KNN model on historical price data
    and uses it to generate buy/sell signals based on the probability that the next
    price bar will be higher.
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
        Initialise the LorentzianStrategy.

        Parameters
        ----------
        params : dict | None, optional
            Optional configuration dictionary. Recognised keys:
            - ``k`` (int): Number of neighbours for KNN (default 8).
            - ``lookback`` (int): Look‑back window size (default 2000).
            - ``subsample`` (int): Sub‑sampling factor for incremental updates (default 4).
        """
        super().__init__(params)
        self.k = params.get("k", 8) if params else 8
        self.lookback = params.get("lookback", 2000) if params else 2000
        self.subsample = params.get("subsample", 4) if params else 4
        self._model: Optional[LorentzianKNN] = None

    def _get_or_build_model(self, df: pd.DataFrame) -> LorentzianKNN:
        """
        Lazily construct or retrieve the in‑memory Lorentzian KNN model.

        Parameters
        ----------
        df : pd.DataFrame
            Historical price data containing at least a ``close`` column.

        Returns
        -------
        LorentzianKNN
            The fitted KNN model ready for inference.
        """
        if self._model is None:
            self._model = LorentzianKNN(k=self.k, lookback=self.lookback, subsample=self.subsample)
            feat_df = compute_lorentzian_features(df)
            features = feat_df[LORENTZIAN_FEATURES].fillna(0).values
            # Label: 1 if price goes up next bar
            labels = (df["close"].shift(-1) > df["close"]).astype(int).values
            self._model.fit_library(features[:-1], labels[:-1])
        return self._model

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        """
        Analyse the most recent data slice and generate a trading signal.

        Parameters
        ----------
        data : pd.DataFrame
            Recent price data for the target symbol.
        symbol : str
            Ticker symbol for which the signal is generated.

        Returns
        -------
        Signal | None
            A populated :class:`Signal` object if confidence exceeds the threshold,
            otherwise ``None``.
        """
        # Need at least two bars to apply a confirmation filter
        if len(data) < 51:
            return None

        model = self._get_or_build_model(data)

        feat_df = compute_lorentzian_features(data)
        features = feat_df[LORENTZIAN_FEATURES].fillna(0).values

        import torch

        # Current bar probability
        latest_features = features[-1:].astype(np.float32)
        x_curr = torch.tensor(latest_features, dtype=torch.float32)
        prob_curr = float(model.forward(x_curr).item())

        # Previous bar probability for confirmation
        prev_features = features[-2:-1].astype(np.float32)
        x_prev = torch.tensor(prev_features, dtype=torch.float32)
        prob_prev = float(model.forward(x_prev).item())

        confidence = abs(prob_curr - 0.5) * 2
        if confidence < self.confidence_threshold:
            return None

        entry_margin = self.confidence_threshold / 2

        # Simple momentum filter based on price direction of the latest bar
        price_up = data["close"].iloc[-1] > data["open"].iloc[-1]
        price_down = data["close"].iloc[-1] < data["open"].iloc[-1]

        # Long entry: probability above midpoint with upward trend and confirmation
        long_condition = (
            prob_curr > 0.5 + entry_margin
            and prob_curr > prob_prev
            and price_up
        )
        # Short entry: probability below midpoint with downward trend and confirmation
        short_condition = (
            prob_curr < 0.5 - entry_margin
            and prob_curr < prob_prev
            and price_down
        )

        if long_condition:
            side = "buy"
        elif short_condition:
            side = "sell"
        else:
            return None

        return Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={"lorentzian_prob": round(prob_curr, 4), "k": self.k},
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        """
        Perform a walk‑forward backtest on the supplied DataFrame.

        The model is trained on the first half of the data and then used to predict
        the second half, updating the library incrementally according to ``subsample``.

        Parameters
        ----------
        df : pd.DataFrame
            Historical price data with a ``close`` column.

        Returns
        -------
        BacktestSignals
            Container with boolean Series for entry/exit points for both long and short
            positions.
        """
        model = LorentzianKNN(k=self.k, lookback=self.lookback, subsample=self.subsample)
        feat_df = compute_lorentzian_features(df)
        features = feat_df[LORENTZIAN_FEATURES].fillna(0).values
        labels = (df["close"].shift(-1) > df["close"]).astype(int).fillna(0).values

        split = len(features) // 2
        model.fit_library(features[:split], labels[:split])

        import torch

        probs = np.zeros(len(df))
        for i in range(split, len(features)):
            x = torch.tensor(features[i : i + 1], dtype=torch.float32)
            probs[i] = float(model.forward(x).item())

            if i % self.subsample == 0 and i + 1 < len(features):
                model._library_X = torch.cat(
                    [model._library_X, torch.tensor(features[i : i + 1], dtype=torch.float32)]
                )
                model._library_y = torch.cat(
                    [model._library_y, torch.tensor([labels[i]], dtype=torch.float32)]
                )

        prob_series = pd.Series(probs, index=df.index).shift(1)

        entry_margin = self.confidence_threshold / 2

        # Tightened entry: require two consecutive bars satisfying the margin
        long_entries = (
            (prob_series > 0.5 + entry_margin)
            & (prob_series.shift(1) > 0.5 + entry_margin)
        )
        short_entries = (
            (prob_series < 0.5 - entry_margin)
            & (prob_series.shift(1) < 0.5 - entry_margin)
        )

        # Exit logic: soften the threshold to allow earlier exits while avoiding whipsaws
        long_exits = prob_series < 0.5 + entry_margin / 2
        short_exits = prob_series > 0.5 - entry_margin / 2

        return BacktestSignals(
            entries=long_entries.fillna(False),
            exits=long_exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )