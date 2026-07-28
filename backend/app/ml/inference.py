"""
Unified ML inference service. Loaded once at app startup.
Provides ensemble predictions for any symbol.
"""
import time
import pandas as pd
from typing import Any, Optional
from app.ml.features.engineer import engineer_features, create_sequences, FEATURE_COLS
from app.ml.features.normalization import FeatureScaler
from app.config import settings
from pathlib import Path
import structlog

logger = structlog.get_logger()

_inference_service: "InferenceService | None" = None


class InferenceService:
    def __init__(self):
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, FeatureScaler] = {}
        self.weights = {"lstm": 0.50, "xgboost": 0.35, "lorentzian": 0.15}

    def load_models(self) -> None:
        """Load all active model artifacts from disk."""
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
        from app.ml.models.gemini_signal import get_gemini_engine
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
        """Returns True if at least one model (lstm, xgboost, lorentzian, gemini) is loaded."""
        return any(
            self.models.get(k) is not None
            for k in ("lstm", "xgboost", "lorentzian", "gemini")
        )

    async def predict(self, data: Optional[pd.DataFrame], symbol: Optional[str]) -> dict | None:
        """
        Generate ensemble prediction for the latest bar in data.
        Returns: {prediction: 'up'|'down'|'neutral', confidence: float, ...}
        """
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            logger.warning("Predict called with invalid or empty data", symbol=symbol)
            return None
        if not symbol:
            logger.warning("Predict called with missing symbol")
            return None
        if not self.models:
            logger.warning("No models loaded; inference disabled")
            return None

        start_time = time.perf_counter()
        try:
            # Feature engineering
            feat_df = engineer_features(data, normalize=False)
            if feat_df is None or len(feat_df) < 60:
                logger.warning("Feature engineering yielded insufficient data", symbol=symbol)
                return None

            # Gather individual predictions
            predictions: dict[str, float] = {}

            if "lstm" in self.models:
                scaler = self.scalers.get("default")
                feat_df_norm = feat_df.copy()
                if scaler:
                    feat_df_norm[FEATURE_COLS] = scaler.transform(feat_df_norm[FEATURE_COLS])
                X, _ = create_sequences(feat_df_norm, seq_len=60)
                if X is not None and len(X) > 0:
                    import torch
                    prob = float(self.models["lstm"].predict_proba(X[-1:]).item())
                    predictions["lstm"] = prob
                else:
                    logger.debug("LSTM sequence generation produced no data", symbol=symbol)

            if "xgboost" in self.models:
                import numpy as np
                X_flat = feat_df[FEATURE_COLS].values[-1:]
                if X_flat.size > 0:
                    prob = float(self.models["xgboost"].predict_proba(X_flat)[0])
                    predictions["xgboost"] = prob
                else:
                    logger.debug("XGBoost feature slice empty", symbol=symbol)

            if "lorentzian" in self.models:
                from app.ml.models.lorentzian_knn import compute_lorentzian_features, LORENTZIAN_FEATURES
                import torch, numpy as np
                lf = compute_lorentzian_features(data)
                if lf is not None and not lf.empty:
                    last_row = lf[LORENTZIAN_FEATURES].fillna(0).values[-1:]
                    if last_row.size > 0:
                        x = torch.tensor(last_row, dtype=torch.float32)
                        prob = float(self.models["lorentzian"].forward(x).item())
                        predictions["lorentzian"] = prob
                    else:
                        logger.debug("Lorentzian feature vector empty", symbol=symbol)
                else:
                    logger.debug("Lorentzian features computation failed or empty", symbol=symbol)

            # Gemini signal (async)
            if "gemini" in self.models:
                try:
                    gemini_prob = await self.models["gemini"].predict_proba(
                        data, symbol, interval="1d"
                    )
                    if gemini_prob is not None:
                        predictions["gemini"] = gemini_prob
                except Exception as e:
                    logger.warning("Gemini model prediction failed", error=str(e), symbol=symbol)

            if not predictions:
                logger.warning("No predictions generated from any model", symbol=symbol)
                return None

            # Weighted ensemble
            total_w = sum(self.weights.get(n, 1.0) for n in predictions)
            if total_w == 0:
                logger.error("Total weight for ensemble is zero", symbol=symbol)
                return None
            ensemble_prob = sum(v * self.weights.get(n, 1.0) for n, v in predictions.items()) / total_w
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
            logger.info(
                "Inference completed",
                symbol=symbol,
                signal_count=len(predictions),
                execution_time=exec_time,
                pnl=None,
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
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
        _inference_service.load_models()
    return _inference_service