"""
Lorentzian KNN strategy — Python port of TradingView's most popular ML indicator.
Uses Lorentzian distance (robust to outliers) for k-nearest-neighbors classification.
"""
import pandas as pd
import numpy as np
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.models.lorentzian_knn import LorentzianKNN, compute_lorentzian_features, LORENTZIAN_FEATURES


class LorentzianStrategy(AbstractStrategy):
    name = "lorentzian_knn"
    display_name = "Lorentzian Classification (ML)"
    market_type = "equity"
    strategy_type = "ml_enhanced"
    risk_bucket = "directional"
    tick_interval_seconds = 300.0
    confidence_threshold = 0.65

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.k = params.get("k", 8) if params else 8
        self.lookback = params.get("lookback", 2000) if params else 2000
        self.subsample = params.get("subsample", 4) if params else 4
        self._model: LorentzianKNN | None = None

    def _get_or_build_model(self, df: pd.DataFrame) -> LorentzianKNN:
        """Build KNN library from historical data (lazy, in-memory)."""
        if self._model is None:
            self._model = LorentzianKNN(k=self.k, lookback=self.lookback, subsample=self.subsample)
            feat_df = compute_lorentzian_features(df)
            features = feat_df[LORENTZIAN_FEATURES].fillna(0).values
            # Label: 1 if price goes up next bar
            labels = (df["close"].shift(-1) > df["close"]).astype(int).values
            self._model.fit_library(features[:-1], labels[:-1])
        return self._model

    async def analyze(self, data: pd.DataFrame, symbol: str) -> Signal | None:
        if len(data) < 50:
            return None
        model = self._get_or_build_model(data)
        feat_df = compute_lorentzian_features(data)
        latest_features = feat_df[LORENTZIAN_FEATURES].fillna(0).values[-1:]

        import torch
        x = torch.tensor(latest_features, dtype=torch.float32)
        prob = float(model.forward(x).item())
        confidence = abs(prob - 0.5) * 2

        if confidence < self.confidence_threshold:
            return None

        side = "buy" if prob > 0.5 else "sell"
        return Signal(
            symbol=symbol, side=side, confidence=confidence,
            strategy_name=self.name, strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={"lorentzian_prob": round(prob, 4), "k": self.k},
        )

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        model = LorentzianKNN(k=self.k, lookback=self.lookback, subsample=self.subsample)
        feat_df = compute_lorentzian_features(df)
        features = feat_df[LORENTZIAN_FEATURES].fillna(0).values
        labels = (df["close"].shift(-1) > df["close"]).astype(int).fillna(0).values

        # Build library on first half, predict on second half (walk-forward)
        split = len(features) // 2
        model.fit_library(features[:split], labels[:split])

        import torch
        probs = np.zeros(len(df))
        for i in range(split, len(features)):
            x = torch.tensor(features[i:i+1], dtype=torch.float32)
            probs[i] = float(model.forward(x).item())
            # Update library incrementally
            if i % self.subsample == 0 and i + 1 < len(features):
                model._library_X = torch.cat([model._library_X, torch.tensor(features[i:i+1], dtype=torch.float32)])
                model._library_y = torch.cat([model._library_y, torch.tensor([labels[i]], dtype=torch.float32)])

        prob_series = pd.Series(probs, index=df.index).shift(1)
        entries = prob_series > 0.5 + self.confidence_threshold / 2
        exits = prob_series < 0.5
        short_entries = prob_series < 0.5 - self.confidence_threshold / 2
        short_exits = prob_series > 0.5
        return BacktestSignals(
            entries=entries.fillna(False), exits=exits.fillna(False),
            short_entries=short_entries.fillna(False), short_exits=short_exits.fillna(False),
        )
