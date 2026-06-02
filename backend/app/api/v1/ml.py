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
