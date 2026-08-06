"""
Unified ML inference service. Loaded once at app startup.
Provides ensemble predictions for any symbol.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import structlog

from app.config import settings
from app.ml.features.engineer import FEATURE_COLS, create_sequences, engineer_features
from app.ml.features.normalization import FeatureScaler
from app.ml.models.gemini_signal import get_gemini_engine

logger = structlog.get_logger()

_inference_service: Optional["InferenceService"] = None


class InferenceService:
    """
    Service that loads ML model artifacts and provides ensemble predictions.

    The service loads LSTM, XGBoost, Lorentzian KNN, and Gemini signal models
    (when an API key is configured). It also loads a feature scaler used by
    the LSTM model. Models are weighted to produce a single probability
    which is then translated into an up/down/neutral signal.
    """

    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, FeatureScaler] = {}
        self.weights: Dict[str, float] = {
            "lstm": 0.50,
            "xgboost": 0.35,
            "lorentzian": 0.15,
        }

    def load_models(self) -> None:
        """
        Load all active model artifacts from disk.

        The method looks for model files in the directory specified by
        ``settings.models_dir``. If a file is found, the corresponding model
        class is imported and the model is loaded. Missing files are ignored,
        and any loading errors are logged. The Gemini signal engine is always
        attempted; if it reports availability, it is added to the model set
        and the ensemble weights are re‑scaled accordingly.
        """
        models_dir = Path(settings.models_dir)
        if not models_dir.exists():
            logger.warning("models_artifacts directory not found — ML predictions disabled")
            return

        # Try loading LSTM
        lstm_path = models_dir / "lstm_latest.pt"
        if lstm_path.exists():
            try:
                from app.ml.models.lstm import LSTMPredictor

                self.models["lstm"] = LSTMPredictor.load(str(lstm_path))
                logger.info("LSTM model loaded", path=str(lstm_path))
            except Exception as e:
                logger.error("Failed to load LSTM", error=str(e))

        # Try loading XGBoost
        xgb_path = models_dir / "xgboost_latest.ubj"
        if xgb_path.exists():
            try:
                from app.ml.models.xgboost_model import XGBoostClassifier

                self.models["xgboost"] = XGBoostClassifier.load(str(xgb_path))
                logger.info("XGBoost model loaded")
            except Exception as e:
                logger.error("Failed to load XGBoost", error=str(e))

        # Try loading Lorentzian KNN
        lk_path = models_dir / "lorentzian_latest.pkl"
        if lk_path.exists():
            try:
                from app.ml.models.lorentzian_knn import LorentzianKNN

                self.models["lorentzian"] = LorentzianKNN.load(str(lk_path))
                logger.info("Lorentzian KNN loaded")
            except Exception as e:
                logger.error("Failed to load Lorentzian KNN", error=str(e))

        # Load scaler
        scaler_path = models_dir / "scaler_latest.pkl"
        if scaler_path.exists():
            self.scalers["default"] = FeatureScaler.load(str(scaler_path))

        # Load Gemini signal engine (always available when API key is set)
        gemini = get_gemini_engine()
        if gemini.is_available:
            self.models["gemini"] = gemini
            self.weights["gemini"] = 0.20
            # Reduce other weights proportionally
            total = sum(v for k, v in self.weights.items() if k != "gemini")
            scale = 0.80 / total if total > 0 else 1.0
            for k in list(self.weights.keys()):
                if k != "gemini":
                    self.weights[k] = round(self.weights[k] * scale, 3)
            logger.info("Gemini signal engine loaded", weight=0.20)

    def has_any_model(self) -> bool:
        """
        Return ``True`` if at least one model (LSTM, XGBoost, Lorentzian, Gemini)
        is successfully loaded.
        """
        return any(
            self.models.get(k) is not None
            for k in ("lstm", "xgboost", "lorentzian", "gemini")
        )

    async def predict(self, data: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Generate an ensemble prediction for the latest bar in ``data``.

        Parameters
        ----------
        data: pd.DataFrame
            Raw market data containing at least the columns required by the
            feature engineering pipeline.
        symbol: str
            Ticker symbol for which the prediction is being made.

        Returns
        -------
        dict | None
            A dictionary with keys ``prediction``, ``probability``, ``confidence``,
            and ``individual`` (per‑model probabilities). Returns ``None`` if no
            models are loaded, the data is insufficient, or an error occurs.
        """
        if not self.models:
            return None

        logger.info("Inference started", symbol=symbol, data_points=len(data))
        start_time = time.perf_counter()
        try:
            # Feature engineering
            feat_df = engineer_features(data, normalize=False)
            if len(feat_df) < 60:
                return None

            # Gather individual predictions
            predictions: Dict[str, float] = {}

            if "lstm" in self.models:
                scaler = self.scalers.get("default")
                if scaler:
                    feat_df_norm = feat_df.copy()
                    feat_df_norm[FEATURE_COLS] = scaler.transform(feat_df_norm[FEATURE_COLS])
                    X, _ = create_sequences(feat_df_norm, seq_len=60)
                    if len(X) > 0:
                        import torch

                        prob = float(self.models["lstm"].predict_proba(X[-1:]).item())
                        predictions["lstm"] = prob

            if "xgboost" in self.models:
                import numpy as np

                X_flat = feat_df[FEATURE_COLS].values[-1:]
                prob = float(self.models["xgboost"].predict_proba(X_flat)[0])
                predictions["xgboost"] = prob

            if "lorentzian" in self.models:
                from app.ml.models.lorentzian_knn import LORENTZIAN_FEATURES, compute_lorentzian_features

                import torch
                import numpy as np

                lf = compute_lorentzian_features(data)
                x = torch.tensor(
                    lf[LORENTZIAN_FEATURES].fillna(0).values[-1:], dtype=torch.float32
                )
                prob = float(self.models["lorentzian"].forward(x).item())
                predictions["lorentzian"] = prob

            # Gemini signal (async)
            if "gemini" in self.models:
                try:
                    gemini_prob = await self.models["gemini"].predict_proba(
                        data, symbol, interval="1d"
                    )
                    if gemini_prob is not None:
                        predictions["gemini"] = gemini_prob
                except Exception as e:
                    logger.warning("Gemini model prediction failed", error=str(e))

            if not predictions:
                return None

            # Weighted ensemble
            total_w = sum(self.weights.get(n, 1.0) for n in predictions)
            ensemble_prob = sum(
                v * self.weights.get(n, 1.0) for n, v in predictions.items()
            ) / total_w
            confidence = abs(ensemble_prob - 0.5) * 2

            if ensemble_prob > 0.5 + 0.05:
                prediction = "up"
            elif ensemble_prob < 0.5 - 0.05:
                prediction = "down"
            else:
                prediction = "neutral"

            result = {
                "prediction": prediction,
                "probability": round(ensemble_prob, 4),
                "confidence": round(confidence, 4),
                "individual": {k: round(v, 4) for k, v in predictions.items()},
            }

            exec_time = time.perf_counter() - start_time
            # Estimate P&L as a simple function of ensemble probability deviation from neutrality
            estimated_pnl = (ensemble_prob - 0.5) * 2

            logger.info(
                "Inference completed",
                symbol=symbol,
                signal_count=len(predictions),
                execution_time=exec_time,
                pnl=estimated_pnl,
                result=result,
            )
            return result
        except Exception as e:
            exec_time = time.perf_counter() - start_time
            logger.error(
                "Inference error",
                symbol=symbol,
                error=str(e),
                execution_time=exec_time,
            )
            return None


def get_inference_service() -> InferenceService:
    """
    Retrieve a singleton instance of :class:`InferenceService`.

    The service is instantiated on first call and its models are loaded
    immediately. Subsequent calls return the cached instance.
    """
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
        _inference_service.load_models()
    return _inference_service