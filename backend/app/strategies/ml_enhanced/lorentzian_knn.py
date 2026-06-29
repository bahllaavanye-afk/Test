"""
Lorentzian KNN strategy — Python port of TradingView's most popular ML indicator.
Uses Lorentzian distance (robust to outliers) for k-nearest-neighbors classification.
"""
import logging
import time
import pandas as pd
import numpy as np
from app.strategies.base import AbstractStrategy, Signal, BacktestSignals
from app.ml.models.lorentzian_knn import LorentzianKNN, compute_lorentzian_features, LORENTZIAN_FEATURES

logger = logging.getLogger(__name__)


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
        self._signal_counter = 0

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
        start_time = time.perf_counter()
        if len(data) < 50:
            logger.info(
                "analyze_skipped",
                extra={"symbol": symbol, "reason": "insufficient_data",
                       "execution_time_ms": (time.perf_counter() - start_time) * 1000},
            )
            return None

        model = self._get_or_build_model(data)
        feat_df = compute_lorentzian_features(data)
        features = feat_df[LORENTZIAN_FEATURES].fillna(0).values

        # Latest feature vector
        latest_features = features[-1:].astype(np.float32)
        # Previous feature vector for confirmation (if available)
        prev_features = features[-2:-1].astype(np.float32) if len(features) >= 2 else None

        import torch

        x_latest = torch.tensor(latest_features, dtype=torch.float32)
        prob = float(model.forward(x_latest).item())
        confidence = abs(prob - 0.5) * 2

        if confidence < self.confidence_threshold:
            logger.info(
                "analyze_no_signal",
                extra={"symbol": symbol, "confidence": confidence,
                       "threshold": self.confidence_threshold,
                       "execution_time_ms": (time.perf_counter() - start_time) * 1000},
            )
            return None

        # Confirmation filter: previous probability must agree with direction
        if prev_features is not None:
            x_prev = torch.tensor(prev_features, dtype=torch.float32)
            prev_prob = float(model.forward(x_prev).item())
            direction_consistent = (prob > 0.5 and prev_prob > 0.5) or (prob < 0.5 and prev_prob < 0.5)
            if not direction_consistent:
                logger.info(
                    "analyze_no_signal",
                    extra={"symbol": symbol, "reason": "direction_not_confirmed",
                           "execution_time_ms": (time.perf_counter() - start_time) * 1000},
                )
                return None

        # Price filter: align with 20‑period SMA
        sma20 = data["close"].rolling(window=20).mean().iloc[-1]
        price = data["close"].iloc[-1]
        if np.isnan(sma20):
            logger.info(
                "analyze_no_signal",
                extra={"symbol": symbol, "reason": "sma_not_available",
                       "execution_time_ms": (time.perf_counter() - start_time) * 1000},
            )
            return None

        if prob > 0.5:
            if price <= sma20:
                logger.info(
                    "analyze_no_signal",
                    extra={"symbol": symbol, "reason": "price_below_sma",
                           "execution_time_ms": (time.perf_counter() - start_time) * 1000},
                )
                return None
        else:
            if price >= sma20:
                logger.info(
                    "analyze_no_signal",
                    extra={"symbol": symbol, "reason": "price_above_sma",
                           "execution_time_ms": (time.perf_counter() - start_time) * 1000},
                )
                return None

        side = "buy" if prob > 0.5 else "sell"
        signal = Signal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            risk_bucket=self.risk_bucket,
            metadata={"lorentzian_prob": round(prob, 4), "k": self.k},
        )
        self._signal_counter += 1
        logger.info(
            "signal_generated",
            extra={"symbol": symbol, "side": side, "confidence": confidence,
                   "lorentzian_prob": round(prob, 4), "signal_count": self._signal_counter,
                   "execution_time_ms": (time.perf_counter() - start_time) * 1000},
        )
        return signal

    def backtest_signals(self, df: pd.DataFrame) -> BacktestSignals:
        start_time = time.perf_counter()
        model = LorentzianKNN(k=self.k, lookback=self.lookback, subsample=self.subsample)
        feat_df = compute_lorentzian_features(df)
        features = feat_df[LORENTZIAN_FEATURES].fillna(0).values
        labels = (df["close"].shift(-1) > df["close"]).astype(int).fillna(0).values

        # Walk‑forward split
        split = len(features) // 2
        model.fit_library(features[:split], labels[:split])

        import torch

        probs = np.zeros(len(df))
        for i in range(split, len(features)):
            x = torch.tensor(features[i : i + 1], dtype=torch.float32)
            probs[i] = float(model.forward(x).item())

            # Incrementally update library
            if i % self.subsample == 0 and i + 1 < len(features):
                model._library_X = torch.cat(
                    [model._library_X, torch.tensor(features[i : i + 1], dtype=torch.float32)]
                )
                model._library_y = torch.cat(
                    [model._library_y, torch.tensor([labels[i]], dtype=torch.float32)]
                )

        prob_series = pd.Series(probs, index=df.index).shift(1)

        # SMA20 filter
        sma20 = df["close"].rolling(window=20).mean()
        price = df["close"]

        # Tightened entry conditions
        long_entries = (
            (prob_series > 0.5 + self.confidence_threshold) &
            (price > sma20) &
            (prob_series.shift(1) > 0.5)
        )
        short_entries = (
            (prob_series < 0.5 - self.confidence_threshold) &
            (price < sma20) &
            (prob_series.shift(1) < 0.5)
        )

        # Exit conditions: probability reverts to neutral or price crosses SMA opposite direction
        long_exits = (prob_series < 0.5) | (price < sma20)
        short_exits = (prob_series > 0.5) | (price > sma20)

        # Simple P&L approximation
        returns = df["close"].pct_change().fillna(0)
        position = pd.Series(0, index=df.index)
        position[long_entries] = 1
        position[short_entries] = -1
        position = position.ffill().fillna(0)

        # Apply exits by resetting position when exit signals occur
        position[long_exits] = 0
        position[short_exits] = 0
        position = position.ffill().fillna(0)

        pnl = (position.shift(1) * returns).sum()

        signal_count = int(
            long_entries.sum() + short_entries.sum() +
            long_exits.sum() + short_exits.sum()
        )
        logger.info(
            "backtest_completed",
            extra={"signal_count": signal_count, "pnl": pnl,
                   "execution_time_ms": (time.perf_counter() - start_time) * 1000},
        )

        return BacktestSignals(
            entries=long_entries.fillna(False),
            exits=long_exits.fillna(False),
            short_entries=short_entries.fillna(False),
            short_exits=short_exits.fillna(False),
        )