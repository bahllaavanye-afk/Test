"""ML model management and prediction endpoints."""
from fastapi import APIRouter, Depends
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
