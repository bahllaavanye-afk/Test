"""Real-time price WebSocket endpoint."""
import logging
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, validator

from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Wildcard topic used for subscribers that want all symbols
PRICES_ALL_TOPIC = "prices:*"


class PriceUpdate(BaseModel):
    """Schema representing a price update for a given symbol."""

    symbol: str = Field(
        ...,
        description="Ticker symbol of the instrument.",
        example="AAPL",
    )
    price: float = Field(
        ...,
        ge=0,
        description="Latest price of the instrument. Must be non‑negative.",
        example=172.45,
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of the price update.",
        example="2024-01-01T12:30:00Z",
    )

    @validator("timestamp")
    def timestamp_not_future(cls, v: datetime) -> datetime:
        """Ensure the timestamp is not in the future."""
        now = datetime.utcnow()
        if v > now:
            raise ValueError("timestamp cannot be in the future")
        return v

    class Config:
        schema_extra = {
            "example": {
                "symbol": "AAPL",
                "price": 172.45,
                "timestamp": "2024-01-01T12:30:00Z",
            }
        }


@router.websocket("/ws/prices")
async def prices_ws_all(websocket: WebSocket):
    """Subscribe to all price updates across all symbols."""
    await manager.connect(websocket, PRICES_ALL_TOPIC)
    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
            except Exception as exc:
                logger.warning("prices_ws_all receive error: %s", exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, PRICES_ALL_TOPIC)


@router.websocket("/ws/prices/{symbol}")
async def prices_ws(websocket: WebSocket, symbol: str):
    topic = f"prices:{symbol}"
    await manager.connect(websocket, topic)
    try:
        while True:
            try:
                await websocket.receive_text()  # keep alive / ping handling
            except Exception as exc:
                logger.warning("prices_ws receive error for %s: %s", symbol, exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, topic)