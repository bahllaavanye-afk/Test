"""
Unified ML inference service. Loaded once at app startup.
Provides ensemble predictions for any symbol.
"""
import pandas as pd
from typing import Any
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

        # Try loading SSM (Mamba-inspired state space model)
        ssm_path = models_dir / "ssm_latest.pt"
        if ssm_path.exists():
            try:
                from app.ml.models.ssm_model import SSMPredictor
                self.models["ssm"] = SSMPredictor.load(str(ssm_path))
                self.weights["ssm"] = 0.10
                logger.info("SSM model loaded")
            except Exception as e:
                logger.warning("Failed to load SSM", error=str(e))

        # Try loading PatchTST (Transformer with patching — SOTA on S&P, NASDAQ)
        patchtst_path = models_dir / "patchtst_latest.pt"
        if patchtst_path.exists():
            try:
                from app.ml.models.patch_tst import PatchTST
                self.models["patchtst"] = PatchTST.load(str(patchtst_path))
                self.weights["patchtst"] = 0.10
                logger.info("PatchTST model loaded")
            except Exception as e:
                logger.warning("Failed to load PatchTST", error=str(e))

        # Try loading N-BEATS (stable extended-horizon forecasting)
        nbeats_path = models_dir / "nbeats_latest.pt"
        if nbeats_path.exists():
            try:
                from app.ml.models.nbeats_model import NBeatsModel
                self.models["nbeats"] = NBeatsModel.load(str(nbeats_path))
                self.weights["nbeats"] = 0.10
                logger.info("N-BEATS model loaded")
            except Exception as e:
                logger.warning("Failed to load N-BEATS", error=str(e))

        # Load Gemini signal engine (always available when API key is set)
        from app.ml.models.gemini_signal import get_gemini_engine
        gemini = get_gemini_engine()
        if gemini.is_available:
            self.models["gemini"] = gemini
            self.weights["gemini"] = 0.20
            logger.info("Gemini signal engine loaded", weight=0.20)

        # Renormalise weights to sum to 1.0
        total_w = sum(self.weights.get(k, 0.0) for k in self.models)
        if total_w > 0:
            for k in list(self.weights.keys()):
                if k in self.models:
                    self.weights[k] = round(self.weights[k] / total_w, 4)

    def has_any_model(self) -> bool:
        """Returns True if at least one model (lstm, xgboost, lorentzian, gemini) is loaded."""
        return any(
            self.models.get(k) is not None
            for k in ("lstm", "xgboost", "lorentzian", "gemini")
        )

    async def predict(self, data: pd.DataFrame, symbol: str) -> dict | None:
        """
        Generate ensemble prediction for the latest bar in data.
        Returns: {prediction: 'up'|'down'|'neutral', confidence: float, ...}
        """
        if not self.models:
            return None

        try:
            # Feature engineering
            feat_df = engineer_features(data, normalize=False)
            if len(feat_df) < 60:
                return None

            # Gather individual predictions
            predictions = {}

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
                from app.ml.models.lorentzian_knn import compute_lorentzian_features, LORENTZIAN_FEATURES
                import torch, numpy as np
                lf = compute_lorentzian_features(data)
                x = torch.tensor(lf[LORENTZIAN_FEATURES].fillna(0).values[-1:], dtype=torch.float32)
                prob = float(self.models["lorentzian"].forward(x).item())
                predictions["lorentzian"] = prob

            # SSM (Mamba-inspired state space model)
            if "ssm" in self.models:
                try:
                    scaler = self.scalers.get("default")
                    if scaler:
                        feat_df_norm = feat_df.copy()
                        feat_df_norm[FEATURE_COLS] = scaler.transform(feat_df_norm[FEATURE_COLS])
                        X, _ = create_sequences(feat_df_norm, seq_len=60)
                        if len(X) > 0:
                            import torch
                            out = self.models["ssm"].forward(X[-1:])
                            predictions["ssm"] = float(out.item() if hasattr(out, "item") else out[0])
                except Exception as e:
                    logger.debug("SSM prediction failed", error=str(e))

            # PatchTST (patched transformer — low MSE on financial data 2024)
            if "patchtst" in self.models:
                try:
                    scaler = self.scalers.get("default")
                    if scaler:
                        feat_df_norm = feat_df.copy()
                        feat_df_norm[FEATURE_COLS] = scaler.transform(feat_df_norm[FEATURE_COLS])
                        X, _ = create_sequences(feat_df_norm, seq_len=60)
                        if len(X) > 0:
                            import torch
                            out = self.models["patchtst"].forward(X[-1:])
                            predictions["patchtst"] = float(out.item() if hasattr(out, "item") else out[0])
                except Exception as e:
                    logger.debug("PatchTST prediction failed", error=str(e))

            # N-BEATS (stable extended-horizon forecasting)
            if "nbeats" in self.models:
                try:
                    import numpy as np
                    X_flat = feat_df[FEATURE_COLS].values[-1:]
                    out = self.models["nbeats"].predict_proba(X_flat)
                    predictions["nbeats"] = float(out[0] if hasattr(out, "__len__") else out)
                except Exception as e:
                    logger.debug("N-BEATS prediction failed", error=str(e))

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
            ensemble_prob = sum(v * self.weights.get(n, 1.0) for n, v in predictions.items()) / total_w
            confidence = abs(ensemble_prob - 0.5) * 2

            if ensemble_prob > 0.5 + 0.05:
                prediction = "up"
            elif ensemble_prob < 0.5 - 0.05:
                prediction = "down"
            else:
                prediction = "neutral"

            return {
                "prediction": prediction,
                "probability": round(ensemble_prob, 4),
                "confidence": round(confidence, 4),
                "individual": {k: round(v, 4) for k, v in predictions.items()},
            }
        except Exception as e:
            logger.error("Inference error", symbol=symbol, error=str(e))
            return None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
        _inference_service.load_models()
    return _inference_service
