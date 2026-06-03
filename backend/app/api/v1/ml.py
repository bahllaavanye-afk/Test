"""ML model management and prediction endpoints."""
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.api.deps import get_current_user
from app.models.ml_model import MLModel
from app.models.user import User
from app.utils.logging import logger
from pydantic import BaseModel, ConfigDict
from datetime import datetime

router = APIRouter(prefix="/ml", tags=["ml"])


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
        logger.warning("list_models DB query failed", error=str(exc))
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
        logger.warning("list_signals DB query failed", error=str(exc))
        return []


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
    symbol: str = "SPY"
    n_splits: int = 5
    lookback_days: int = 365


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
            detail="No trained models. Run POST /api/v1/experiments/train first.",
        )

    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np

        ticker = yf.Ticker(req.symbol)
        df = ticker.history(
            period=f"{req.lookback_days + 30}d", interval="1d", progress=False
        )
        if df.empty or len(df) < 60:
            raise HTTPException(422, f"Not enough data for {req.symbol}")
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
                    idx = actual_returns.index[-len(probs):]
                    returns_by_model["lstm"] = pd.Series(probs - 0.5, index=idx)
            except Exception:
                pass

        if "xgboost" in inference.models:
            try:
                X_flat = feat_df[FEATURE_COLS].values
                probs = inference.models["xgboost"].predict_proba(X_flat)
                idx = actual_returns.index[-len(probs):]
                returns_by_model["xgboost"] = pd.Series(probs - 0.5, index=idx)
            except Exception:
                pass

        if "lorentzian" in inference.models:
            try:
                from app.ml.models.lorentzian_knn import (
                    compute_lorentzian_features, LORENTZIAN_FEATURES,
                )
                import torch
                lf = compute_lorentzian_features(df)
                X_lk = torch.tensor(
                    lf[LORENTZIAN_FEATURES].fillna(0).values, dtype=torch.float32
                )
                with torch.no_grad():
                    probs = inference.models["lorentzian"].forward(X_lk).squeeze().numpy()
                idx = actual_returns.index[-len(probs):]
                returns_by_model["lorentzian"] = pd.Series(probs - 0.5, index=idx)
            except Exception:
                pass

        if len(returns_by_model) < 2:
            raise HTTPException(
                422,
                "Need at least 2 trained models to optimise. "
                f"Currently available: {list(inference.models.keys())}",
            )

        ensemble = EnsembleModel()
        optimal_weights = ensemble.optimize_weights_walk_forward(
            returns_by_model, actual_returns, n_splits=req.n_splits
        )

        # Update live inference weights
        for name, w in optimal_weights.items():
            if name in inference.weights:
                inference.weights[name] = round(w, 4)

        return {
            "symbol": req.symbol,
            "n_splits": req.n_splits,
            "models_used": list(returns_by_model.keys()),
            "optimal_weights": optimal_weights,
            "previous_weights": {k: inference.weights.get(k) for k in optimal_weights},
            "message": "Ensemble weights updated in-memory. Restart to persist.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("optimize_ensemble_weights failed", error=str(exc))
        raise HTTPException(500, str(exc))
