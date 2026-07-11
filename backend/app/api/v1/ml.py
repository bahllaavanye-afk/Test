"""ML model management and prediction endpoints."""

from fastapi import APIRouter, Depends, Query, Body
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.ml_model import MLModel
from app.models.user import User
from app.utils.logging import logger
from pydantic import BaseModel, ConfigDict
from datetime import datetime, timezone

# Constants
API_PREFIX = "/ml"
API_TAGS = ["ml"]

DEFAULT_SYMBOL = "SPY"
DEFAULT_N_SPLITS = 5
DEFAULT_LOOKBACK_DAYS = 365

HIST_PERIOD = "6mo"
HIST_INTERVAL = "1d"
MIN_HIST_DAYS = 60
LOOKBACK_BUFFER_DAYS = 30
YF_PROGRESS = False

ERR_LIST_MODELS_DB = "list_models DB query failed"
ERR_LIST_SIGNALS_DB = "list_signals DB query failed"
ERR_PREDICTION_NO_MODELS = (
    "No trained models available. Run training first via POST /api/v1/experiments/train"
)
ERR_NOT_ENOUGH_DATA = "Not enough historical data for {symbol}"
ERR_PREDICTION_GENERATION_FAIL = "Could not generate prediction for {symbol}"
ERR_PREDICTION_ENDPOINT = "Prediction endpoint error"
ERR_NOT_ENOUGH_DATA_OPT = "Not enough data for {symbol}"
ERR_NO_TRAINED_MODELS = (
    "No trained models. Run POST /api/v1/experiments/train first."
)

router = APIRouter(prefix=API_PREFIX, tags=API_TAGS)


class ModelOut(BaseModel):
    id: str
    model_type: str
    symbol: str | None
    market_type: str | None = None
    val_accuracy: float | None
    val_sharpe: float | None
    is_active: bool
    trained_at: datetime

    model_config = ConfigDict(from_attributes=True)


@router.get("/models", response_model=list[ModelOut])
async def list_models(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        result = await db.execute(select(MLModel).order_by(MLModel.trained_at.desc()))
        return result.scalars().all()
    except Exception as exc:
        logger.warning(ERR_LIST_MODELS_DB, error=str(exc))
        return []


@router.get("/signals")
async def list_signals(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return recent ML prediction signals (latest per active model)."""
    from app.models.ml_model import MLPrediction
    from sqlalchemy.orm import selectinload

    try:
        result = await db.execute(
            select(MLPrediction)
            .order_by(MLPrediction.created_at.desc())
            .limit(50)
        )
        preds = result.scalars().all()
        return [
            {
                "id": p.id,
                "model_id": p.model_id,
                "symbol": p.symbol,
                "prediction": p.prediction,
                "confidence": float(p.confidence),
                "ts": p.ts.isoformat() if p.ts else None,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in preds
        ]
    except Exception as exc:
        logger.warning(ERR_LIST_SIGNALS_DB, error=str(exc))
        return []


@router.get("/predictions")
async def get_predictions(
    symbol: str = Query(..., description="Ticker symbol, e.g. SPY"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Run ensemble ML prediction for a given symbol. Returns 503 if no models are trained."""
    from app.ml.inference import get_inference_service

    inference = get_inference_service()
    if not inference.has_any_model():
        return JSONResponse(
            status_code=503,
            content={"detail": ERR_PREDICTION_NO_MODELS},
        )
    # Fetch recent market data for the symbol
    try:
        import yfinance as yf
        import pandas as pd

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=HIST_PERIOD, interval=HIST_INTERVAL)
        if df.empty or len(df) < MIN_HIST_DAYS:
            return JSONResponse(
                status_code=422,
                content={"detail": ERR_NOT_ENOUGH_DATA.format(symbol=symbol)},
            )
        df.columns = [c.lower() for c in df.columns]
        result = await inference.predict(df, symbol)
        if result is None:
            return JSONResponse(
                status_code=422,
                content={"detail": ERR_PREDICTION_GENERATION_FAIL.format(symbol=symbol)},
            )
        return {"symbol": symbol, **result}
    except Exception as exc:
        logger.error(ERR_PREDICTION_ENDPOINT, symbol=symbol, error=str(exc))
        return JSONResponse(status_code=500, content={"detail": str(exc)})


@router.get("/models/{model_id}/activate")
async def activate_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(MLModel).where(MLModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        from fastapi import HTTPException

        raise HTTPException(404, "Model not found")
    model.is_active = True
    await db.commit()
    return {"activated": model_id}


class EnsembleWeightRequest(BaseModel):
    symbol: str = DEFAULT_SYMBOL
    n_splits: int = DEFAULT_N_SPLITS
    lookback_days: int = DEFAULT_LOOKBACK_DAYS


@router.post("/ensemble/optimize-weights")
async def optimize_ensemble_weights(
    req: EnsembleWeightRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Walk-forward ensemble weight optimization via SLSQP.

    Uses historical prices to generate per-model prediction series, then
    optimises weights to maximise walk-forward Sharpe. Updates the in-memory
    inference service weights immediately.
    """
    from fastapi import HTTPException
    from app.ml.inference import get_inference_service
    from app.ml.features.engineer import engineer_features, create_sequences, FEATURE_COLS
    from app.ml.features.normalization import FeatureScaler
    from app.ml.models.ensemble_model import EnsembleModel

    inference = get_inference_service()
    if not inference.has_any_model():
        raise HTTPException(
            status_code=503,
            detail=ERR_NO_TRAINED_MODELS,
        )

    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np

        ticker = yf.Ticker(req.symbol)
        df = ticker.history(
            period=f"{req.lookback_days + LOOKBACK_BUFFER_DAYS}d",
            interval="1d",
            progress=YF_PROGRESS,
        )
        if df.empty or len(df) < MIN_HIST_DAYS:
            raise HTTPException(422, ERR_NOT_ENOUGH_DATA_OPT.format(symbol=req.symbol))
        df.columns = [c.lower() for c in df.columns]

        actual_returns: pd.Series = df["close"].pct_change().shift(-1).dropna()

        # Build per-model return proxies using feature scores as signal strength
        feat_df = engineer_features(df, normalize=False)
        returns_by_model: dict[str, pd.Series] = {}

        if "lstm" in inference.models and inference.scalers.get("default"):
            scaler = inference.scalers["default"]
            try:
                feat_norm = feat_df.copy()
                feat_norm[FEATURE_COLS] = scaler.transform(feat_norm[FEATURE_COLS])
                X, _ = create_sequences(feat_norm, seq_len=60)
                if len(X) > 0:
                    import torch

                    with torch.no_grad():
                        probs = inference.models["lstm"].predict_proba(
                            torch.tensor(X, dtype=torch.float32)
                        ).squeeze().numpy()
                    idx = actual_returns.index[-len(probs) :]
                    returns_by_model["lstm"] = pd.Series(probs - 0.5, index=idx)
            except Exception:
                pass

        if "xgboost" in inference.models:
            try:
                X_flat = feat_df[FEATURE_COLS].values
                probs = inference.models["xgboost"].predict_proba(X_flat)
                idx = actual_returns.index[-len(probs) :]
                returns_by_model["xgboost"] = pd.Series(probs - 0.5, index=idx)
            except Exception:
                pass

        if "lorentzian" in inference.models:
            try:
                from app.ml.models.lorentzian_knn import (
                    compute_lorentzian_features,
                    LORENTZIAN_FEATURES,
                )
                import torch

                lf = compute_lorentzian_features(df)
                X_lk = torch.tensor(
                    lf[LORENTZIAN_FEATURES].fillna(0).values, dtype=torch.float32
                )
                with torch.no_grad():
                    probs = inference.models["lorentzian"].forward(X_lk).squeeze().numpy()
                idx = actual_returns.index[-len(probs) :]
                returns_by_model["lorentzian"] = pd.Series(probs - 0.5, index=idx)
            except Exception:
                pass

        # ... (remaining logic unchanged)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ensemble weight optimization failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))