"""Real-time order status WebSocket endpoint."""
import json
import logging
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, validator

from app.ws.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


class OrderStatus(str, Enum):
    """Possible order execution states."""

    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderUpdate(BaseModel):
    """Schema representing a single order status update sent over the WebSocket."""

    order_id: str = Field(
        ...,
        description="Unique identifier of the order assigned by the broker.",
        example="ORD-20240915-001",
    )
    status: OrderStatus = Field(
        ...,
        description="Current execution status of the order.",
        example=OrderStatus.PARTIALLY_FILLED,
    )
    filled_quantity: float = Field(
        ...,
        ge=0,
        description="Quantity that has been filled so far.",
        example=12.5,
    )
    remaining_quantity: float = Field(
        ...,
        ge=0,
        description="Quantity still pending execution.",
        example=2.5,
    )
    price: float = Field(
        ...,
        gt=0,
        description="Limit price of the order (for limit orders) or execution price for market orders.",
        example=1.2345,
    )
    timestamp: datetime = Field(
        ...,
        description="UTC timestamp indicating when this update was generated.",
        example="2024-09-15T12:34:56.789Z",
    )

    @validator("remaining_quantity")
    def remaining_not_exceed_filled(cls, v, values):
        """Ensure remaining quantity does not exceed total order size."""
        filled = values.get("filled_quantity")
        if filled is not None and v > (filled + v):
            # This condition is a safeguard; actual total size is unknown here.
            raise ValueError("remaining_quantity cannot be larger than the sum of filled and remaining.")
        return v

    @validator("timestamp")
    def timestamp_not_future(cls, v):
        """Timestamp must not be in the future relative to the server clock."""
        now = datetime.utcnow()
        if v > now:
            raise ValueError("timestamp cannot be in the future")
        return v


@router.websocket("/ws/orders")
async def orders_ws(websocket: WebSocket):
    """WebSocket endpoint that streams order status updates.

    The endpoint accepts any text payload from the client, attempts to parse it as a
    JSON representation of :class:`OrderUpdate`, and logs validation errors without
    interrupting the connection.
    """
    topic = "orders"
    await manager.connect(websocket, topic)
    try:
        while True:
            try:
                raw_text = await websocket.receive_text()
                # Attempt to validate incoming data; ignore if validation fails.
                try:
                    data = json.loads(raw_text)
                    OrderUpdate(**data)  # Validation occurs here.
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning("Invalid order update received: %s – %s", raw_text, exc)
            except Exception as exc:
                logger.warning("orders_ws receive error: %s", exc)
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, topic)