"""
Unified ML inference service. Loaded once at app startup.
Provides ensemble predictions for any symbol.
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

import pandas as pd
import structlog
from pydantic import BaseModel, Field, validator

from app.ml.features.engineer import FEATURE_COLS, create_sequences, engineer_features
from app.ml.features.normalization import FeatureScaler
from app.config import settings

logger = structlog.get_logger()

_inference_service: "InferenceService | None" = None


class PredictionRequest(BaseModel):
    """
    Schema for a prediction request.

    Attributes
    ----------
    symbol: str
        Ticker symbol for which the prediction is requested.
    data: List[Dict[str, Any]]
        List of raw bar data dictionaries (e.g., OHLCV). Each dict should contain
        the same keys required by the feature engineering step.
    """

    symbol: str = Field(
        ...,
        description="Ticker symbol for the prediction request",
        example="AAPL",
    )
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Raw bar data as a list of dictionaries (e.g., OHLCV rows)",
        example=[
            {"timestamp": "2024-01-01T09:30:00Z", "open": 150.0, "high": 152.0, "low": 149.5, "close": 151.0, "volume": 1000000},
            {"timestamp": "2024-01-01T09:31:00Z", "open": 151.0, "high": 152.5, "low": 150.8, "close": 152.0, "volume": 800000},
        ],
    )

    @validator("symbol")
    def symbol_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("symbol must be a non-empty string")
        return v

    @validator("data")
    def data_must_be_non_empty(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not v:
            raise ValueError("data list must contain at least one bar")
        return v


class PredictionResponse(BaseModel):
    """
    Schema for a prediction response.

    Attributes
    ----------
    prediction: Literal['up', 'down', 'neutral']
        Directional prediction.
    probability: float
        Ensemble probability (0.0‑1.0) indicating confidence in the prediction.
    confidence: float
        Normalized confidence metric (0.0‑1.0).
    individual: Dict[str, float]
        Mapping of model names to their individual probabilities.
    execution_time: float
        Time taken for inference in seconds.
    pnl: float
        Estimated profit & loss based on the ensemble probability.
    """

    prediction: Literal["up", "down", "neutral"] = Field(
        ...,
        description="Directional prediction based on ensemble probability",
        example="up",
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ensemble probability (0.0‑1.0)",
        example=0.73,
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized confidence metric (0.0‑1.0)",
        example=0.46,
    )
    individual: Dict[str, float] = Field(
        ...,
        description="Individual model probabilities",
        example={"lstm": 0.71, "xgboost": 0.68, "lorentzian": 0.70},
    )
    execution_time: float = Field(
        ...,
        description="Inference execution time in seconds",
        example=0.0123,
    )
    pnl: float = Field(
        ...,
        description="Estimated profit & loss based on probability deviation from neutrality",
        example=0.46,
    )

    @validator("probability", "confidence")
    def probability_bounds(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("value must be between 0.0 and 1.0")
        return v


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

    async def predict(self, data: pd.DataFrame, symbol: str) -> dict | None:
        """
        Generate ensemble prediction for the latest bar in data.
        Returns: {prediction: 'up'|'down'|'neutral', confidence: float, ...}
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
                from app.ml.models.lorentzian_knn import LORENTZIAN_FEATURES, compute_lorentzian_features

                import torch, numpy as np

                lf = compute_lorentzian_features(data)
                x = torch.tensor(lf[LORENTZIAN_FEATURES].fillna(0).values[-1:], dtype=torch.float32)
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
            # Extend result to match PredictionResponse schema
            result.update(
                {
                    "execution_time": exec_time,
                    "pnl": estimated_pnl,
                }
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